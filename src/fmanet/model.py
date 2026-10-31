"""FMANet: fused multi-level attention features with a constraint fusion network.

The architecture is ported from the original notebooks (``notebooks/``) and
checked against the parameter count printed by their ``model.summary()``.

Pipeline: three stacked 3x3 conv blocks form a *low-level* stream; three more
conv blocks continue from it to form a *high-level* stream. Each stream is
refined by a dilated residual-inception block. The two streams are then pooled,
embedded by shared dense layers, and combined with **constraint fusion**: the
weight of each stream is its (tanh) cosine similarity to the concatenated
features, normalised so the two weights sum to one.
"""

from __future__ import annotations

import keras
from keras import Model, layers, ops

BN_EPSILON = 1.1e-5
INCEPTION_FILTERS = 192


# ---------------------------------------------------------------------------
# Constraint fusion layers (replace the notebook's Lambda layers)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="fmanet")
class CosineSimilarity(layers.Layer):
    """tanh of the cosine similarity of two ``(batch, d)`` tensors -> ``(batch, 1)``."""

    def call(self, inputs):
        a, b = inputs
        dot_ab = ops.sum(a * b, axis=1, keepdims=True)
        norm = ops.sqrt(
            ops.sum(a * a, axis=1, keepdims=True) * ops.sum(b * b, axis=1, keepdims=True)
        )
        return ops.tanh(dot_ab / ops.maximum(norm, keras.config.epsilon()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


@keras.saving.register_keras_serializable(package="fmanet")
class ConstraintWeights(layers.Layer):
    """Normalise two similarity scores into weights that sum to one."""

    def call(self, inputs):
        alpha_1, alpha_2 = inputs
        total = alpha_1 + alpha_2
        return alpha_1 / total, alpha_2 / total

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def conv_bn(x, filters: int, kernel_size: int = 3, dilation: int = 1):
    """Conv (ReLU, same padding) followed by batch normalisation."""
    x = layers.Conv2D(
        filters, kernel_size, dilation_rate=dilation, padding="same", activation="relu"
    )(x)
    return layers.BatchNormalization(axis=-1, epsilon=BN_EPSILON)(x)


def dilated_residual_inception(x, filters: int = INCEPTION_FILTERS):
    """Dilated (rates 1, 2, 3) multiplicative residual-inception block."""
    a, b, c = (conv_bn(x, filters, 3, d) for d in (1, 2, 3))
    m1, m2, m3 = layers.Multiply()([a, b]), layers.Multiply()([a, c]), layers.Multiply()([b, c])
    a2, b2, c2 = conv_bn(m1, filters, 3, 1), conv_bn(m2, filters, 3, 2), conv_bn(m3, filters, 3, 3)
    fused = layers.Multiply()([a2, b2, c2])
    fused = conv_bn(fused, filters, 1)
    return layers.Add()([fused, x])


def residual_spatial_attention(x, reduction_ratio: int = 16):
    """Spatial attention map from 1x1 convs, applied multiplicatively with a residual add."""
    channels = x.shape[-1]
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(x)
    h = layers.Conv2D(channels // reduction_ratio, 1, padding="same", activation="relu")(h)
    h = layers.Conv2D(channels, 1, padding="same", activation="relu")(h)
    attention_map = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(h)
    return layers.Add()([layers.Multiply()([attention_map, x]), x])


def residual_channel_attention(x, gamma: int = 2):
    """Squeeze-and-excitation style channel attention with a residual add."""
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(channels // gamma, activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Add()([layers.Multiply()([x, s]), x])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def build_fmanet(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 8,
    use_attention_modules: bool = False,
) -> Model:
    """Build FMANet.

    Args:
        input_shape: Input image shape.
        num_classes: 8 for the Kather 5,000-tile dataset, 9 for CRC-VAL-HE-7K.
        use_attention_modules: In the original notebooks the spatial- and
            channel-attention modules are created but their outputs are *not*
            connected to the classifier (``x_l = A1; x_g = A2``), so the trained
            graph does not contain them. ``False`` (default) reproduces that
            graph exactly (5,389,832 parameters for 8 classes). ``True`` routes
            the low-level stream through spatial attention and the high-level
            stream through channel attention, as shown in the architecture figure.
            That variant was **not** trained in the original notebooks.
    """
    inputs = layers.Input(input_shape)

    # Low-level stream
    x1 = conv_bn(inputs, 64)
    x2 = conv_bn(x1, 64)
    x3 = conv_bn(x2, 64)
    low = dilated_residual_inception(layers.Concatenate()([x1, x2, x3]))

    # High-level stream (continues from the low-level features)
    x4 = conv_bn(x3, 64)
    x5 = conv_bn(x4, 64)
    x6 = conv_bn(x5, 64)
    high = dilated_residual_inception(layers.Concatenate()([x4, x5, x6]))

    if use_attention_modules:
        low = residual_spatial_attention(low)
        high = residual_channel_attention(high)

    combined = layers.GlobalAveragePooling2D()(layers.Concatenate()([low, high]))
    low = layers.Dense(384, activation="relu")(layers.GlobalAveragePooling2D()(low))
    high = layers.Dense(384, activation="relu")(layers.GlobalAveragePooling2D()(high))

    shared_1 = layers.Dense(768, activation="relu")
    shared_2 = layers.Dense(768, activation="relu")
    low, high, combined = (shared_2(shared_1(t)) for t in (low, high, combined))

    alpha_low, alpha_high = ConstraintWeights()(
        [CosineSimilarity()([low, combined]), CosineSimilarity()([high, combined])]
    )
    fused = layers.Concatenate()(
        [layers.Multiply()([alpha_low, low]), layers.Multiply()([alpha_high, high])]
    )
    fused = layers.Dense(64, activation="relu")(fused)
    outputs = layers.Dense(num_classes, activation="softmax")(fused)
    return Model(inputs, outputs, name="fmanet")


def compile_fmanet(model: Model, learning_rate: float | None = None) -> Model:
    """Compile with the optimiser and loss used in the original notebooks (Nadam, lr 1e-4)."""
    model.compile(
        optimizer=keras.optimizers.Nadam(
            learning_rate=1e-4 if learning_rate is None else learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

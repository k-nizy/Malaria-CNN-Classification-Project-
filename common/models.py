"""
common/models.py — the group's hand-built models.

Model 1: CustomResNet — assignment §3.1
  - residual CNN built with the Keras Subclassing API (custom Layer/Model
    subclasses with explicit call()), NOT a pretrained ResNet,
  - residual design follows He et al. (2016): identity shortcuts where shapes
    match, 1x1 projection shortcuts where downsampling changes shapes,
  - trained from scratch on the malaria dataset,
  - Grad-CAM ready: the last conv block's output is stashed on
    self._last_conv_features during every forward pass (assignment §5.4 note).

Saving/loading: subclassed models are saved weights-only (.weights.h5) and
rebuilt with CustomResNet() + load_weights — see save()/load() helpers below.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(layers.Layer):
    """
    Basic residual block (He et al. 2016, §3.2):
        out = ReLU( F(x) + shortcut(x) )
    where F = [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] and shortcut is either
    identity (stride 1, channels match) or a 1x1 Conv+BN projection.
    """

    def __init__(self, filters: int, stride: int = 1, l2: float = 0.0,
                 bn_momentum: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.l2 = l2
        self.bn_momentum = bn_momentum   # experiment lever: 0.99 (default) vs 0.9

        reg = regularizers.l2(l2) if l2 and l2 > 0 else None
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same",
                                   kernel_regularizer=reg, name="conv_a")
        self.bn1 = layers.BatchNormalization(momentum=bn_momentum, name="bn_a")
        self.conv2 = layers.Conv2D(filters, 3, padding="same",
                                   kernel_regularizer=reg, name="conv_b")
        self.bn2 = layers.BatchNormalization(momentum=bn_momentum, name="bn_b")
        self.proj = None

    def build(self, input_shape):
        """
        Projection shortcut is needed whenever the shortcut path must change
        shape: downsampling (stride != 1) OR channel-width change (He et al.
        2016, §3.2 — option B). Decided here because input channels are only
        known at build time. Caught by tests: stage1 goes 32->64 ch at
        stride 1 and crashed with a shape mismatch before this fix.
        """
        in_ch = int(input_shape[-1])
        if self.stride != 1 or in_ch != self.filters:
            reg = regularizers.l2(self.l2) if self.l2 and self.l2 > 0 else None
            self.proj = keras.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.stride,
                              use_bias=False, kernel_regularizer=reg),
                layers.BatchNormalization(momentum=self.bn_momentum),
            ], name="projection")
        super().build(input_shape)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        shortcut = self.proj(x, training=training) if self.proj is not None else x
        return tf.nn.relu(out + shortcut)

    def get_config(self):
        return {**super().get_config(),
                "filters": self.filters, "stride": self.stride, "l2": self.l2,
                "bn_momentum": self.bn_momentum}


class CustomResNet(keras.Model):
    """
    Small from-scratch ResNet for 128x128 malaria cell crops
    (scales to any input size — all convs are 'same'-padded).

    Stem:  Conv3x3/32 -> BN -> ReLU
    Stage1: 2 x ResidualBlock(64)
    Stage2: ResidualBlock(128, stride 2) + ResidualBlock(128)
    Stage3: ResidualBlock(256, stride 2) + ResidualBlock(256)   <- Grad-CAM tap
    Head:  GAP -> Dense(256) -> Dropout -> Dense(1, sigmoid)

    width_mult: channel width multiplier (experiment lever: 0.5/1.0/1.5)
    blocks_per_stage: depth lever (experiment lever: 2 vs 3)
    """

    def __init__(self, dropout: float = 0.3, l2: float = 0.0,
                 width_mult: float = 1.0, blocks_per_stage: int = 2,
                 bn_momentum: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout
        self.l2 = l2
        self.width_mult = width_mult
        self.blocks_per_stage = blocks_per_stage
        self.bn_momentum = bn_momentum   # experiment lever: 0.99 (default) vs 0.9

        w = lambda f: max(8, int(f * width_mult))
        reg = regularizers.l2(l2) if l2 and l2 > 0 else None

        self.stem = keras.Sequential([
            layers.Conv2D(w(32), 3, padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(momentum=bn_momentum),
        ], name="stem")

        def stage(filters, stride):
            blocks = [ResidualBlock(w(filters), stride=stride, l2=l2,
                                    bn_momentum=bn_momentum)]
            blocks += [ResidualBlock(w(filters), l2=l2,
                                     bn_momentum=bn_momentum)
                       for _ in range(blocks_per_stage - 1)]
            return blocks

        self.stage1 = stage(64, 1)
        self.stage2 = stage(128, 2)
        self.stage3 = stage(256, 2)

        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout)
        self.head = layers.Dense(1, activation="sigmoid", name="classifier")

        # Grad-CAM tap (assignment §5.4): populated during call()
        self._last_conv_features = None

    def call(self, x, training=False):
        x = tf.nn.relu(self.stem(x, training=training))
        for blk in self.stage1:
            x = blk(x, training=training)
        for blk in self.stage2:
            x = blk(x, training=training)
        for blk in self.stage3:
            x = blk(x, training=training)
        self._last_conv_features = x            # (B, H/4, W/4, 256*w) — tap
        x = self.gap(x)
        x = self.dropout(x, training=training)
        return self.head(x)

    # subclassed-model saving: weights only (see module docstring)
    def save_weights_only(self, path: str):
        self.save_weights(path)

    def get_config(self):
        return {"dropout": self.dropout_rate, "l2": self.l2,
                "width_mult": self.width_mult,
                "blocks_per_stage": self.blocks_per_stage,
                "bn_momentum": self.bn_momentum}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_custom_resnet(img_size: tuple[int, int] = (128, 128),
                        dropout: float = 0.3, l2: float = 0.0,
                        width_mult: float = 1.0,
                        blocks_per_stage: int = 2,
                        bn_momentum: float = 0.99) -> CustomResNet:
    """
    Build the model and create its weights via a dummy forward call.

    Note: Keras 3's model.build() does NOT reliably build sub-layer weights on
    subclassed models (ResidualBlocks build lazily on first call) — a dummy
    call is the canonical way, and makes count_params() correct.
    """
    model = CustomResNet(dropout=dropout, l2=l2, width_mult=width_mult,
                         blocks_per_stage=blocks_per_stage,
                         bn_momentum=bn_momentum)
    model(tf.zeros((1, img_size[0], img_size[1], 3)), training=False)
    return model


# ---------------------------------------------------------------------------
# Transfer-learning backbones (Models 2 & 3)
# ---------------------------------------------------------------------------

def build_transfer_model(backbone_name: str,
                         img_size: tuple[int, int] = (128, 128),
                         dropout: float = 0.3,
                         dense_units: int = 256,
                         freeze_backbone: bool = True,
                         l2: float = 0.0) -> tf.keras.Model:
    """
    Functional transfer-learning model: pretrained backbone (ImageNet) +
    GAP + Dense head. Backbones bring their own preprocessing layer so the
    pipeline stays raw-RGB end to end (avoids double-preprocessing bugs).

    freeze_backbone=True gives the exp_01 frozen baseline; unfreezing for
    fine-tuning experiments is done afterwards in the notebooks by setting
    layer.trainable on the top-N layers (see unfreeze_top_n).
    """
    img_input = layers.Input(shape=(img_size[0], img_size[1], 3))
    name = backbone_name.lower()

    if name in ("mobilenetv2", "mobilenet"):
        from tensorflow.keras.applications import MobileNetV2
        prep = layers.Lambda(
            lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t),
            name="mobilenet_preprocess")
        base = MobileNetV2(include_top=False, weights="imagenet",
                           input_shape=(img_size[0], img_size[1], 3))
    elif name in ("densenet121", "densenet"):
        from tensorflow.keras.applications import DenseNet121
        prep = layers.Lambda(
            lambda t: tf.keras.applications.densenet.preprocess_input(t),
            name="densenet_preprocess")
        base = DenseNet121(include_top=False, weights="imagenet",
                           input_shape=(img_size[0], img_size[1], 3))
    elif name in ("efficientnetb0", "efficientnet"):
        from tensorflow.keras.applications import EfficientNetB0
        prep = layers.Lambda(
            lambda t: tf.keras.applications.efficientnet.preprocess_input(t),
            name="efficientnet_preprocess")
        base = EfficientNetB0(include_top=False, weights="imagenet",
                              input_shape=(img_size[0], img_size[1], 3))
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = not freeze_backbone
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None

    y = prep(img_input)
    y = base(y, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(dropout)(y)
    out = layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=reg, name="classifier")(y)

    model = tf.keras.Model(img_input, out, name=f"{base.name}_malaria")
    model._malaria_backbone = base          # handle for unfreezing in notebooks
    return model


def unfreeze_top_n(model: tf.keras.Model, n: int = 20) -> None:
    """Unfreeze the last n layers of the backbone for fine-tuning experiments."""
    base = getattr(model, "_malaria_backbone", model)
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers[-n:]:
        layer.trainable = True

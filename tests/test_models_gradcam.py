"""
tests/test_models_gradcam.py — verification for common/models.py + gradcam.py

The critical tests for the assignment's two technical traps:
  §3.1  ResNet must be subclassing-API, train from scratch, and hit high
        accuracy on a learnable task (synthetic blobs stand in for the real
        dataset — same 1=parasitized/0=uninfected convention).
  §5.4  Grad-CAM must work on the subclassed model AND actually localize the
        blob — not just produce a heatmap-shaped array. A heatmap that lights
        up the blob region is real evidence the tap + math are correct.
Also covers: weight save/load round-trip, get_config (subclassing requirement),
config levers (width_mult/blocks_per_stage), projection shortcuts, transfer
model builder, and the Sequential/functional Grad-CAM path.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.data import SEED, build_datasets, make_synthetic_cell_dataset  # noqa: E402
from common.models import (  # noqa: E402
    CustomResNet,
    ResidualBlock,
    build_custom_resnet,
    build_transfer_model,
    unfreeze_top_n,
)
from common.gradcam import (  # noqa: E402
    _find_last_conv_layer,
    make_gradcam_heatmap,
    overlay_heatmap,
    pick_gradcam_examples,
    gradcam_report_grid,
)

tf.keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_resnet(tmp_path_factory):
    """
    Train the real CustomResNet on synthetic blobs.

    A narrow stem (width_mult=0.25) keeps the from-scratch model small enough
    (~120k params) to converge on ~100 tiny images in seconds — the full-width
    2.7M-param version is for the real 27k-image dataset, not a CPU fixture.

    Seed is reset INSIDE the fixture: tests that build models earlier consume
    TF's global RNG, so a file-level seed alone makes the init (and therefore
    convergence) depend on execution order — observed as a real flake.
    """
    tf.keras.utils.set_random_seed(SEED)
    root = make_synthetic_cell_dataset(tmp_path_factory.mktemp("resnet_data"),
                                       per_class=60, img_size=(32, 32), seed=SEED)
    train, val, test = build_datasets(root, img_size=(32, 32), batch_size=16,
                                      seed=SEED)
    model = build_custom_resnet(img_size=(32, 32), dropout=0.2, width_mult=0.25)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train, validation_data=val, epochs=40, verbose=0)
    return model, history, test


def _make_blob_image(seed: int = 7) -> np.ndarray:
    """One 32x32 synthetic 'parasitized' image, same recipe as the dataset."""
    rng = np.random.default_rng(seed)
    img = rng.integers(150, 190, size=(32, 32, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:32, :32]
    img[(yy - 14) ** 2 + (xx - 18) ** 2 <= 25] = (30, 30, 60)
    return img.astype(np.float32)


# ----------------------------------------------------------------------------
# §3.1 Architecture requirements
# ----------------------------------------------------------------------------

def test_resnet_is_subclass_model():
    model = build_custom_resnet(img_size=(32, 32))
    assert isinstance(model, tf.keras.Model) and not isinstance(model, tf.keras.Sequential)
    assert callable(getattr(model, "call", None))


def test_projection_shortcuts_exist():
    """Stride-2 stages must build projection shortcuts (He et al. §3.2)."""
    model = build_custom_resnet(img_size=(32, 32))
    proj_blocks = [blk for stage in (model.stage2, model.stage3)
                   for blk in stage if blk.proj is not None]
    assert len(proj_blocks) >= 2


def test_config_levers_change_architecture():
    small = build_custom_resnet(img_size=(32, 32), width_mult=0.5)
    big = build_custom_resnet(img_size=(32, 32), width_mult=1.0)
    n_small = small.count_params()
    n_big = big.count_params()
    assert n_big > n_small * 1.5          # width lever changes capacity a lot

    deeper = build_custom_resnet(img_size=(32, 32), blocks_per_stage=3)
    assert deeper.count_params() > big.count_params()


def _block_bn_layers(blk):
    """Collect every BatchNormalization inside a stem/stage block.
    ResidualBlock is a custom Layer (no .layers) — use its named sub-layers;
    keras.Sequential (the stem, and a projection shortcut) exposes .layers."""
    if isinstance(blk, tf.keras.layers.BatchNormalization):
        return [blk]
    if hasattr(blk, "bn1"):                      # ResidualBlock
        bns = [blk.bn1, blk.bn2]
        if blk.proj is not None:
            bns += [l for l in blk.proj.layers
                    if isinstance(l, tf.keras.layers.BatchNormalization)]
        return bns
    return [l for l in blk.layers
            if isinstance(l, tf.keras.layers.BatchNormalization)]


def test_bn_momentum_propagates():
    """exp_02 lever: bn_momentum must reach every BatchNorm in the blocks."""
    m = build_custom_resnet(img_size=(32, 32), bn_momentum=0.9)
    for blk in [m.stem, *m.stage1, *m.stage2, *m.stage3]:
        bn_layers = _block_bn_layers(blk)
        assert bn_layers, f"block unexpectedly has no BatchNormalization: {blk.name}"
        for bn in bn_layers:
            assert bn.momentum == 0.9, (
                f"BN momentum not propagated inside {blk.name}"
            )
            assert bn.built is True


def test_get_config_roundtrip():
    """Subclassing requirement (TF guide): get_config/from_config must work."""
    m = CustomResNet(dropout=0.4, l2=1e-4, width_mult=0.5, blocks_per_stage=3)
    cfg = m.get_config()
    m2 = CustomResNet.from_config(cfg)
    assert m2.dropout_rate == 0.4 and m2.width_mult == 0.5
    assert m2.blocks_per_stage == 3


def test_weights_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "resnet.weights.h5")
    m1 = build_custom_resnet(img_size=(32, 32))
    m1.save_weights_only(path)
    m2 = build_custom_resnet(img_size=(32, 32))
    m2.load_weights(path)

    x = _make_blob_image()[None, ...]
    np.testing.assert_allclose(m1(x, training=False).numpy(),
                               m2(x, training=False).numpy(), rtol=1e-5)


# ----------------------------------------------------------------------------
# §3.1 Training from scratch works
# ----------------------------------------------------------------------------

def test_resnet_learns_synthetic_task(trained_resnet):
    model, history, test = trained_resnet
    # training must actually converge on this easy task
    assert history.history["val_accuracy"][-1] > 0.9, \
        f"final val acc={history.history['val_accuracy'][-1]}"
    # eval on held-out test
    ys, probs = [], []
    for x, y in test:
        ys.append(y.numpy().ravel())
        probs.append(model.predict(x, verbose=0).ravel())
    acc = np.mean((np.concatenate(probs) > 0.5) == np.concatenate(ys))
    assert acc > 0.85, f"test acc={acc}"


# ----------------------------------------------------------------------------
# §5.4 Grad-CAM on the subclassed model — THE trap test
# ----------------------------------------------------------------------------

def test_gradcam_tap_populated_on_forward(trained_resnet):
    model, _, _ = trained_resnet
    x = _make_blob_image()[None, ...]
    model(x, training=False)
    assert model._last_conv_features is not None
    # after 2 stride-2 stages from 32x32: 8x8 spatial; width_mult=0.25 -> 64ch
    assert tuple(model._last_conv_features.shape)[1:] == (8, 8, 64)


def test_gradcam_heatmap_localizes_blob(trained_resnet):
    """
    THE §5.4 test: heatmap must be a valid distribution AND concentrate on
    the blob (mean heat inside blob >> outside). This proves the tap + the
    gradient math produce a medically-plausible-style explanation.
    """
    model, _, _ = trained_resnet
    img = _make_blob_image()
    heat, prob = make_gradcam_heatmap(img, model)

    assert heat.shape == (8, 8)
    assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6
    assert prob > 0.5, f"model should call the blob image parasitized (p={prob})"

    # blob center (14,18) in 32x32 -> cell (3-4, 4-5) in 8x8
    inside = heat[3:5, 4:6].mean()
    outside = heat[~np.isin(np.arange(64).reshape(8, 8),
                            np.arange(64).reshape(8, 8)[3:5, 4:6])].mean()
    assert inside > 2 * outside + 0.05, (
        f"heatmap must focus on the blob: inside={inside:.3f} outside={outside:.3f}"
    )


def test_gradcam_both_classes_explicable(trained_resnet):
    """Explain both the predicted class and the other class explicitly."""
    model, _, _ = trained_resnet
    img = _make_blob_image()
    h_pred, _ = make_gradcam_heatmap(img, model, pred_index=None)
    h_pos, _ = make_gradcam_heatmap(img, model, pred_index=1)
    h_neg, _ = make_gradcam_heatmap(img, model, pred_index=0)
    assert h_pred.shape == h_pos.shape == h_neg.shape == (8, 8)
    # explaining class 1 vs class 0 should give different maps (not identical)
    assert not np.allclose(h_pos, h_neg)


def test_gradcam_report_grid(trained_resnet, tmp_path):
    model, _, test = trained_resnet
    # stream a few test images + labels to pick the 3 required example types
    ys, probs, imgs = [], [], []
    for x, y in test:
        imgs.append(x.numpy())
        ys.append(y.numpy().ravel())
        probs.append(model.predict(x, verbose=0).ravel())
    ys = np.concatenate(ys)
    probs = np.concatenate(probs)
    images = np.concatenate(imgs)

    examples = pick_gradcam_examples(ys, probs, images)
    assert len(examples) >= 2       # at least infected + one more type
    fig = gradcam_report_grid(examples[:3], model,
                              save_path=str(tmp_path / "gradcam.png"))
    out = tmp_path / "gradcam.png"
    assert out.exists() and out.stat().st_size > 20_000
    plt.close(fig)


# ----------------------------------------------------------------------------
# Grad-CAM on functional/Sequential models (Models 2 & 3 path)
# ----------------------------------------------------------------------------

def test_gradcam_on_sequential_model():
    """The monkey-patch path: Sequential CNNs get the marker auto-installed."""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(4, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    img = _make_blob_image()
    heat, prob = make_gradcam_heatmap(img, m)
    # conv valid: 32->30, maxpool: ->15, conv valid: ->13
    assert heat.shape == (13, 13)
    assert 0 <= heat.max() <= 1.0 + 1e-6
    # last Conv2D = layers[3] (0=Input, 1=Rescaling, 2=Conv(4), 3=Conv(8))
    assert _find_last_conv_layer(m) is m.layers[3]


def test_transfer_model_builds_and_predicts():
    """Backbone builder works with tiny input; frozen by default."""
    model = build_transfer_model("mobilenetv2", img_size=(96, 96))
    assert not model._malaria_backbone.trainable
    x = np.random.rand(2, 96, 96, 3).astype("float32") * 255
    y = model(x, training=False)
    assert y.shape == (2, 1)
    assert float(tf.reduce_min(y)) >= 0.0 and float(tf.reduce_max(y)) <= 1.0


def test_mobilenetv3small_member_backbone():
    """Member 1's backbone: builds at the V3-minimum 224, frozen by default,
    preprocessing inside the graph, unfreezing works, raw 0-255 in -> sigmoid out."""
    model = build_transfer_model("mobilenetv3small", img_size=(224, 224))
    base = model._malaria_backbone
    assert base.name.lower().startswith("mobilenetv3small")
    assert not base.trainable                      # exp_01 frozen baseline
    n_frozen = model.count_params()
    assert n_frozen > 500_000                      # real pretrained backbone
    # (V3Small minus top = 939,697 params — 'Small' is the point)
    # preprocessing is part of the graph (raw 0-255 RGB in)
    assert any("preprocess" in l.name for l in model.layers)
    x = np.random.rand(2, 224, 224, 3).astype("float32") * 255
    y = model(x, training=False)
    assert y.shape == (2, 1)
    assert float(tf.reduce_min(y)) >= 0.0 and float(tf.reduce_max(y)) <= 1.0
    # fine-tuning lever works (exp_05+): unfreeze top 20, recompile-ready
    unfreeze_top_n(model, n=20)
    assert sum(1 for l in base.layers if l.trainable) >= 20


def test_unfreeze_top_n(tmp_path):
    model = build_transfer_model("mobilenetv2", img_size=(96, 96))
    base = model._malaria_backbone
    n_frozen_before = sum(not l.trainable for l in base.layers)
    unfreeze_top_n(model, n=10)
    n_trainable_after = sum(l.trainable for l in base.layers)
    assert n_trainable_after == 10
    assert n_frozen_before == len(base.layers)

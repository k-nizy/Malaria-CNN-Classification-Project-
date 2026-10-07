"""Smoke tests for Colombe's DenseNet121. No ImageNet download."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from colombe.densenet import (  # noqa: E402
    build_densenet121,
    count_params,
    default_experiments,
    describe_model,
    gradcam_heatmap,
    run_experiment,
)
from common.data import make_synthetic_cell_dataset  # noqa: E402
from common.tb import make_run_name  # noqa: E402


def test_seven_experiments_change_one_thing_at_a_time():
    runs = default_experiments()
    assert len(runs) == 7
    assert [r["id"] for r in runs] == [1, 2, 3, 4, 5, 6, 7]
    names = [make_run_name("densenet121", r["id"], r["name"]) for r in runs]
    assert names[0] == "densenet121_exp_01_baseline_frozen_128"
    assert names[4] == "densenet121_exp_05_unfreeze_conv5"
    assert len(set(names)) == 7
    # batch size is not a hidden variable
    assert {r["batch_size"] for r in runs} == {16}
    # 01 -> 02 is resolution only
    a, b = runs[0], runs[1]
    assert a["img_size"] == (128, 128) and b["img_size"] == (224, 224)
    for key in ("lr", "dropout", "l2", "augment", "finetune", "batch_size"):
        assert a[key] == b[key]
    # 04 -> 05 drops lr AND starts fine-tuning (the fine-tune decision)
    assert runs[3]["finetune"] == "frozen" and runs[4]["finetune"] == "conv5"
    assert runs[4]["lr"] < runs[3]["lr"]
    # 05 -> 06 changes depth only
    assert runs[5]["finetune"] == "conv4"
    assert runs[5]["lr"] == runs[4]["lr"]
    assert runs[5]["dropout"] == runs[4]["dropout"]
    # 07 goes back to conv5 and only changes dropout vs exp 05
    assert runs[6]["finetune"] == "conv5"
    assert runs[6]["dropout"] == 0.4
    assert runs[4]["dropout"] == 0.3


def test_frozen_head_and_conv5_finetune_flags():
    frozen = build_densenet121(img_size=(64, 64), finetune="frozen", weights=None)
    assert count_params(frozen.trainable_weights) == 1025  # 1024-d GAP -> 1 logit + bias
    assert all(not layer.trainable for layer in frozen.backbone.layers)
    assert "frozen_head" in describe_model(frozen)

    frozen.compile(optimizer="adam", loss="binary_crossentropy")
    head_before = frozen.get_layer("logits").kernel.numpy().copy()
    conv = next(layer for layer in frozen.backbone.layers if isinstance(layer, tf.keras.layers.Conv2D))
    conv_before = conv.kernel.numpy().copy()
    x = np.random.randint(0, 255, size=(4, 64, 64, 3)).astype("float32")
    y = np.array([1.0, 0.0, 1.0, 0.0])
    frozen.fit(x, y, epochs=1, verbose=0)
    assert not np.allclose(frozen.get_layer("logits").kernel.numpy(), head_before)
    assert np.allclose(conv.kernel.numpy(), conv_before)

    tuned = build_densenet121(img_size=(64, 64), finetune="conv5", weights=None)
    convs = [layer for layer in tuned.backbone.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    trainable_convs = [layer for layer in convs if layer.trainable]
    assert trainable_convs, "conv5 fine-tune trained no convolutions"
    assert trainable_convs[0].name.startswith("conv5")
    assert any(not layer.trainable for layer in convs)
    bns = [layer for layer in tuned.backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    assert bns and all(not layer.trainable for layer in bns)
    assert tuned.finetune_info["first_tail_layer"].startswith("conv5")
    assert count_params(tuned.trainable_weights) > 1025


def test_forward_and_gradcam_on_random_init():
    model = build_densenet121(img_size=(64, 64), finetune="frozen", weights=None)
    batch = np.random.randint(0, 255, size=(2, 64, 64, 3)).astype("float32")
    probs = model(batch, training=False)
    assert probs.shape == (2, 1)
    assert float(tf.reduce_min(probs)) >= 0.0
    assert float(tf.reduce_max(probs)) <= 1.0

    heat, prob = gradcam_heatmap(model, batch[0], class_index=1)
    assert heat.shape == (2, 2)  # 64 / 32
    assert heat.shape[0] > 0
    assert 0.0 <= float(heat.max()) <= 1.0 + 1e-5
    assert 0.0 <= prob <= 1.0
    heat0, _ = gradcam_heatmap(model, batch[0], class_index=0)
    assert heat0.shape == heat.shape


def test_frozen_step_updates_head_only(tmp_path):
    root = make_synthetic_cell_dataset(tmp_path / "cells", per_class=8, img_size=(64, 64))
    spec = dict(default_experiments()[0])
    spec["img_size"] = (64, 64)
    spec["batch_size"] = 4
    result = run_experiment(
        spec,
        root,
        epochs=1,
        patience=0,
        logdir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        steps_per_epoch=1,
        validation_steps=1,
        weights=None,
        verbose=0,
    )
    assert result["run"] == "densenet121_exp_01_baseline_frozen_128"
    assert Path(result["weights_path"]).exists()
    assert (Path(result["log_dir"]) / "final_metrics.txt").exists()
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "sensitivity"):
        assert key in result["metrics"]
    assert result["history"]["loss"], "training history was not recorded"
    # reload must serve probabilities
    from colombe.densenet import load_trained
    model = load_trained(result, weights=None)
    image = np.random.randint(0, 255, size=(64, 64, 3)).astype("float32")
    prob = float(model(image[None], training=False).numpy().ravel()[0])
    assert 0.0 <= prob <= 1.0

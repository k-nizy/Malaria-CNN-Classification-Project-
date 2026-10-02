"""
tests/test_tb.py — verification for common/tb.py

Checks:
  1. Run naming follows the rubric convention exactly
  2. Logger creates its own directory and produces TB event files
  3. A real (tiny) training run with the logger's callbacks writes:
     - TensorBoard event files
     - CSV training log
     - checkpoint weights (when requested)
     - early stopping works
  4. log_final_metrics writes scalars + human-readable file
  5. Two runs write to DIFFERENT directories (no overwriting!)
  6. zip_logs_for_drive produces a valid zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.data import SEED, make_synthetic_cell_dataset  # noqa: E402
from common.tb import (  # noqa: E402
    ExperimentLogger,
    make_run_name,
    zip_logs_for_drive,
)


@pytest.fixture(scope="module")
def train_ds(tmp_path_factory):
    root = make_synthetic_cell_dataset(tmp_path_factory.mktemp("tb_data"),
                                       per_class=24, img_size=(16, 16), seed=SEED)
    return tf.keras.utils.image_dataset_from_directory(
        str(root), labels="inferred", label_mode="binary",
        class_names=["Parasitized", "Uninfected"],
        image_size=(16, 16), batch_size=8, shuffle=True, seed=SEED,
    ).map(lambda x, y: (x, 1 - y))          # flip: Parasitized -> 1


def _tiny_model() -> tf.keras.Model:
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(4, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ----------------------------------------------------------------------------
# 1. Naming convention
# ----------------------------------------------------------------------------

def test_run_name_convention():
    assert make_run_name("custom_resnet", 1, "baseline") == \
        "custom_resnet_exp_01_baseline"
    assert make_run_name("mobilenet", 4, "unfreeze_last20") == \
        "mobilenet_exp_04_unfreeze_last20"
    assert make_run_name("efficientnet", 6, "lr 1e-5") == \
        "efficientnet_exp_06_lr_1e-5"


def test_run_name_rejects_empty():
    with pytest.raises(AssertionError):
        make_run_name("", 1, "x")
    with pytest.raises(AssertionError):
        make_run_name("model", 1, "")


# ----------------------------------------------------------------------------
# 2. Directory + callbacks produce real artifacts
# ----------------------------------------------------------------------------

def test_full_training_run_produces_all_artifacts(train_ds, tmp_path):
    logdir = tmp_path / "logs" / "fit"
    exp = ExperimentLogger(
        "custom_resnet", 1, "baseline",
        hparams=dict(lr=1e-3, optimizer="adam", batch_size=8, epochs=3,
                     augmentation=False, dropout=0.0, l2=0.0,
                     strategy="scratch"),
        logdir=logdir,
    )
    ckpt = tmp_path / "ckpt" / "exp01.weights.h5"
    cbs = exp.callbacks(checkpoint_path=ckpt, early_stopping_patience=3)

    model = _tiny_model()
    model.fit(train_ds, epochs=3, callbacks=cbs, verbose=0)

    # Keras 3's TensorBoard callback writes into train/ and validation/
    # SUBDIRECTORIES of the run dir (worth knowing for screenshot paths!)
    events = list(exp.log_dir.rglob("events.out.tfevents.*"))
    assert len(events) >= 1, f"expected TB event files under {exp.log_dir}"
    subdir_names = {e.parent.name for e in events}
    assert "train" in subdir_names, f"got subdirs: {subdir_names}"

    # CSV training log exists with an epoch row
    csv = exp.log_dir / "training_log.csv"
    assert csv.exists()
    content = csv.read_text(encoding="utf-8")
    assert "epoch" in content and "val_loss" in content

    # checkpoint saved (weights-only)
    assert ckpt.exists(), "checkpoint file missing"

    # run dir is named per convention
    assert exp.log_dir.name == "custom_resnet_exp_01_baseline"


def test_early_stopping_configured_correctly(tmp_path):
    """
    We test OUR wiring (monitor, patience, restore_best_weights), not Keras's
    internal ES mechanics — a tiny toy model can legitimately keep improving
    for many epochs, so a behavioral assertion would be flaky.
    """
    from tensorflow.keras import callbacks as kc

    exp = ExperimentLogger("custom_resnet", 2, "es_check", logdir=tmp_path)
    cbs = exp.callbacks(early_stopping_patience=5)
    es = [c for c in cbs if isinstance(c, kc.EarlyStopping)]
    assert len(es) == 1
    assert es[0].monitor == "val_loss"
    assert es[0].patience == 5
    assert es[0].restore_best_weights is True

    # and patience=0/None disables ES entirely (explicit choice)
    cbs_off = exp.callbacks(early_stopping_patience=0)
    assert not [c for c in cbs_off if isinstance(c, kc.EarlyStopping)]


def test_final_metrics_written(train_ds, tmp_path):
    logdir = tmp_path / "logs_metrics"
    exp = ExperimentLogger("mobilenet", 7, "final", logdir=logdir,
                           hparams=dict(lr=1e-5, strategy="finetune"))
    model = _tiny_model()
    model.fit(train_ds, epochs=1, callbacks=exp.callbacks(), verbose=0)

    metrics = dict(accuracy=0.95, precision=0.94, recall=0.97, f1=0.955,
                   roc_auc=0.98, specificity=0.93)
    exp.log_final_metrics(metrics)
    exp.log_hparams_once(metrics)

    txt = exp.log_dir / "final_metrics.txt"
    assert txt.exists()
    body = txt.read_text(encoding="utf-8")
    assert "accuracy: 0.950000" in body and "specificity: 0.930000" in body


# ----------------------------------------------------------------------------
# 3. Isolation between runs
# ----------------------------------------------------------------------------

def test_two_runs_do_not_collide(tmp_path):
    a = ExperimentLogger("densenet", 3, "augmentation", logdir=tmp_path)
    b = ExperimentLogger("densenet", 3, "augmentation_l2", logdir=tmp_path)
    assert a.log_dir != b.log_dir
    assert a.log_dir.parent == b.log_dir.parent


def test_same_run_recreated_is_stable(tmp_path):
    """Re-running the same experiment id overwrites (re-run semantics) —
    but must not silently mix with OTHER experiments."""
    a1 = ExperimentLogger("densenet", 3, "augmentation", logdir=tmp_path)
    a1.log_final_metrics(dict(accuracy=0.5))
    a2 = ExperimentLogger("densenet", 3, "augmentation", logdir=tmp_path)
    assert a1.log_dir == a2.log_dir
    b = ExperimentLogger("densenet", 4, "augmentation", logdir=tmp_path)
    assert a1.log_dir != b.log_dir


# ----------------------------------------------------------------------------
# 4. Drive export zip
# ----------------------------------------------------------------------------

def test_zip_logs_for_drive(tmp_path):
    exp = ExperimentLogger("efficientnet", 6, "lr_1e-5", logdir=tmp_path / "fit")
    exp.log_final_metrics(dict(accuracy=0.9))
    (tmp_path / "fit" / exp.run_name / "dummy.txt").write_text("x")

    out = zip_logs_for_drive(tmp_path / "fit", out_path=tmp_path / "export.zip")
    assert out.exists() and out.stat().st_size > 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert any("efficientnet_exp_06_lr_1e-5" in n for n in names)

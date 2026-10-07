"""
common/tb.py — TensorBoard experiment tracking shared by all three models.

Rubric requirements covered here (assignment §4.2):
  - Every experiment logged with: model name, run name, learning rate,
    optimizer, batch size, epochs, augmentation, dropout, L2, transfer
    strategy, train/val curves, and FINAL evaluation metrics in the same run.
  - Descriptive run names: custom_resnet_exp_01_baseline (never test1/final).
  - Run-to-run comparisons possible because every run logs the same scalars
    plus an HPARAMS block that TensorBoard's hparams dashboard can parse.

Usage (in a member notebook):
    from common.tb import ExperimentLogger
    exp = ExperimentLogger(model_name="custom_resnet", exp_id=1,
                           description="baseline",
                           hparams=dict(lr=1e-3, optimizer="adam", batch_size=32,
                                        epochs=30, augmentation=False,
                                        dropout=0.0, l2=0.0, strategy="scratch"))
    model.fit(..., callbacks=exp.callbacks())
    exp.log_final_metrics(metrics_dict)     # acc/prec/recall/f1/auc/specificity
"""

from __future__ import annotations

import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks

# Base log dir — notebooks may override via env var TB_LOGDIR (e.g. Drive path)
DEFAULT_LOGDIR = Path("logs/fit")


def make_run_name(model_name: str, exp_id: int, description: str) -> str:
    """
    Enforce the rubric naming convention:
    custom_resnet_exp_01_baseline / mobilenet_exp_04_unfreeze_last20
    """
    assert model_name and description, "run name parts must be non-empty"
    return f"{model_name}_exp_{exp_id:02d}_{description}".lower().replace(" ", "_")


class ExperimentLogger:
    """One instance per experiment run. Owns its TensorBoard directory."""

    def __init__(self,
                 model_name: str,
                 exp_id: int,
                 description: str,
                 hparams: dict | None = None,
                 logdir: str | Path | None = None,
                 monitor: str = "val_loss"):
        self.run_name = make_run_name(model_name, exp_id, description)
        base = Path(logdir) if logdir else Path(
            __import__("os").environ.get("TB_LOGDIR", DEFAULT_LOGDIR))
        self.log_dir = base / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hparams = hparams or {}
        self.monitor = monitor
        self._hparams_written = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def callbacks(self,
                  checkpoint_path: str | Path | None = None,
                  early_stopping_patience: int = 5) -> list[callbacks.Callback]:
        """
        Standard callback set: TensorBoard (+hparams), CSV logger (extra
        evidence that survives TB issues), optional checkpoint & early stop.
        """
        cb_list: list[callbacks.Callback] = [
            callbacks.TensorBoard(log_dir=str(self.log_dir),
                                  histogram_freq=0,       # keep logs small
                                  write_graph=False,
                                  update_freq="epoch"),
            callbacks.CSVLogger(str(self.log_dir / "training_log.csv"),
                                append=False),
        ]
        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            cb_list.append(callbacks.ModelCheckpoint(
                filepath=str(ckpt), monitor="val_accuracy", save_best_only=True,
                save_weights_only=True, verbose=0))
        if early_stopping_patience and early_stopping_patience > 0:
            cb_list.append(callbacks.EarlyStopping(
                monitor=self.monitor, patience=early_stopping_patience,
                restore_best_weights=True, verbose=0))
        return cb_list

    # ------------------------------------------------------------------
    # Manual scalars (e.g. final test metrics) into the SAME run
    # ------------------------------------------------------------------

    def log_final_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Write final evaluation metrics so run comparisons include them."""
        with tf.summary.create_file_writer(str(self.log_dir)).as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f"final_test/{name}", float(value), step=step)
        # also drop a human-readable copy next to the event files
        lines = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                 for k, v in metrics.items()]
        (self.log_dir / "final_metrics.txt").write_text("\n".join(lines),
                                                        encoding="utf-8")

    def log_hparams_once(self, metrics: dict[str, float] | None = None) -> None:
        """
        Write an HPARAMS record so TensorBoard's hparams dashboard can list
        and compare runs. Call once per run; metrics may be added at the end.
        """
        if self._hparams_written:
            return
        try:
            from tensorboard.plugins.hparams import api as hp
            hparams = {k: v for k, v in self.hparams.items()
                       if isinstance(v, (int, float, str, bool))}
            with tf.summary.create_file_writer(str(self.log_dir)).as_default():
                hp.hparams(hparams)
                if metrics:
                    for name, value in metrics.items():
                        tf.summary.scalar(f"final_test/{name}", float(value),
                                          step=0)
            self._hparams_written = True
        except ImportError:
            # hparams plugin missing in some minimal installs; scalars still work
            pass


def zip_logs_for_drive(logdir: str | Path = DEFAULT_LOGDIR,
                       out_path: str | Path | None = None) -> Path:
    """
    Zip the whole logs/fit tree so it can be copied to Google Drive and later
    shared as the report's open-access appendix link (rubric §4.2 note).
    """
    import shutil
    logdir = Path(logdir)
    out_path = Path(out_path) if out_path else logdir.parent / "tb_logs_export.zip"
    shutil.make_archive(str(out_path.with_suffix("")), "zip", logdir)
    return out_path.with_suffix(".zip")

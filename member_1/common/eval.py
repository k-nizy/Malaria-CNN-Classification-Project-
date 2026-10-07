"""
common/eval.py — metrics & visualization helpers shared by all three models.

Rubric requirements covered here:
  - Per-experiment metrics table: Accuracy, Precision, Recall(sensitivity),
    F1, ROC-AUC, plus Specificity (TNR on uninfected) — reported under BOTH
    naming sets as the assignment requires.
  - The 4 required final-model plots: accuracy curve, loss curve, confusion
    matrix, ROC curve — arranged in a single labeled row per model.
  - Misclassification mining for error analysis (2-3 examples per model).

Label convention (set in common/data.py): 1 = Parasitized (positive/infected),
0 = Uninfected. recall == sensitivity; specificity == TNR on class 0.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use("Agg")                       # headless-safe for tests/CI
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["Parasitized", "Uninfected"]
# Display names indexed BY LABEL VALUE under our clinical convention
# (common.data._flip_label): 0 = Uninfected, 1 = Parasitized. After the flip,
# indexing CLASS_NAMES directly by a label value shows SWAPPED names in every
# figure — caught by tests/test_eval.py::test_display_labels_match_convention.
LABEL_NAMES_BY_VALUE = ["Uninfected", "Parasitized"]


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions(model: tf.keras.Model,
                        dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a batched dataset.
    Returns (y_true, y_prob) where y_prob is P(parasitized) — the sigmoid
    output, NOT the thresholded class. Needed for ROC-AUC.
    """
    ys, probs = [], []
    for x, y in dataset:
        ys.append(y.numpy().ravel())
        probs.append(model.predict(x, verbose=0).ravel())
    return np.concatenate(ys), np.concatenate(probs)


def predictions_from_probs(y_prob: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    """Threshold probabilities into class labels (1=parasitized by default)."""
    return (y_prob >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    specificity: float

    @property
    def sensitivity(self) -> float:
        """Clinical name for recall on the infected class."""
        return self.recall

    def as_row(self) -> dict:
        """Row for the per-experiment metrics table (both naming sets)."""
        d = asdict(self)
        d["sensitivity"] = self.sensitivity
        return d

    def format(self) -> str:
        return (
            f"acc={self.accuracy:.4f}  precision={self.precision:.4f}  "
            f"recall/sensitivity={self.recall:.4f}  specificity={self.specificity:.4f}  "
            f"f1={self.f1:.4f}  roc_auc={self.roc_auc:.4f}"
        )


def evaluate_metrics(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     threshold: float = 0.5) -> Metrics:
    """
    Full metric suite from probabilities.

    WHO naming (assignment §5.1): sensitivity = recall on infected,
    specificity = TNR on uninfected. Both are reported.
    """
    y_pred = predictions_from_probs(y_prob, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    )


def evaluate_model(model: tf.keras.Model,
                   dataset: tf.data.Dataset,
                   threshold: float = 0.5) -> tuple[Metrics, np.ndarray, np.ndarray]:
    """Convenience: collect predictions + compute metrics in one call."""
    y_true, y_prob = collect_predictions(model, dataset)
    return evaluate_metrics(y_true, y_prob, threshold), y_true, y_prob


# ---------------------------------------------------------------------------
# The 4 required plots (final model) — labeled, report-ready
# ---------------------------------------------------------------------------

def plot_training_curves(history: tf.keras.callbacks.History,
                         figsize: tuple[float, float] = (16, 4)):
    """Training vs validation accuracy + loss, side by side (plots 1 & 2 of 4)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    axes[0].plot(epochs, history.history["accuracy"], "o-", label="Training accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], "s-", label="Validation accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.history["loss"], "o-", label="Training loss")
    axes[1].plot(epochs, history.history["val_loss"], "s-", label="Validation loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          normalize: bool = True,
                          figsize: tuple[float, float] = (5, 4)):
    """Confusion matrix (plot 3 of 4). Rows=true, cols=predicted."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fmt = lambda v: f"{v:d}"
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_n = np.nan_to_num(cm_n)
        annot = np.array([[f"{v:.2f}\n({c:d})" for v, c in zip(row_n, row)]
                          for row_n, row in zip(cm_n, cm)])
    else:
        annot = np.array([[str(v) for v in row] for row in cm])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_n if normalize else cm, cmap="Blues")
    ax.set_xticks([0, 1], [f"Pred {n}" for n in LABEL_NAMES_BY_VALUE])
    ax.set_yticks([0, 1], [f"True {n}" for n in LABEL_NAMES_BY_VALUE])
    ax.set_title("Confusion Matrix (1 = Parasitized)")
    for (i, j), txt in np.ndenumerate(annot):
        color = "white" if (cm_n[i, j] if normalize else cm[i, j]) > cm.max() / 2 else "black"
        ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   figsize: tuple[float, float] = (5, 4)):
    """ROC curve with AUC (plot 4 of 4) — the sensitivity/specificity trade-off."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Parasitized vs Uninfected")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def final_evaluation_figure(model: tf.keras.Model,
                            history: tf.keras.callbacks.History,
                            dataset: tf.data.Dataset,
                            threshold: float = 0.5,
                            save_path: str | None = None):
    """
    The rubric's single-row layout: accuracy curve, loss curve, confusion
    matrix, ROC curve — one row per model, labeled and ready to interpret.
    Returns (fig, Metrics, y_true, y_prob).
    """
    metrics, y_true, y_prob = evaluate_model(model, dataset, threshold)
    y_pred = predictions_from_probs(y_prob, threshold)

    fig = plt.figure(figsize=(20, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.001])

    # reuse individual plotters by drawing into sub-axes
    epochs = range(1, len(history.history["accuracy"]) + 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history.history["accuracy"], "o-", label="Train")
    ax1.plot(epochs, history.history["val_accuracy"], "s-", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Curve"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history.history["loss"], "o-", label="Train")
    ax2.plot(epochs, history.history["val_loss"], "s-", label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss Curve"); ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_n = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    im = ax3.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax3.text(j, i, f"{v}\n({cm_n[i, j]:.0%})", ha="center", va="center",
                 color="white" if cm_n[i, j] > 0.5 else "black", fontsize=9)
    ax3.set_xticks([0, 1], [f"Pred {n}" for n in LABEL_NAMES_BY_VALUE], fontsize=8)
    ax3.set_yticks([0, 1], [f"True {n}" for n in LABEL_NAMES_BY_VALUE], fontsize=8)
    ax3.set_title("Confusion Matrix")

    ax4 = fig.add_subplot(gs[0, 3])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax4.plot(fpr, tpr, lw=2, label=f"AUC = {metrics.roc_auc:.4f}")
    ax4.plot([0, 1], [0, 1], "k--", lw=1)
    ax4.set_xlabel("FPR (1-Spec)"); ax4.set_ylabel("TPR (Sens)")
    ax4.set_title("ROC Curve"); ax4.legend(); ax4.grid(alpha=0.3)

    fig.suptitle(
        f"Final evaluation — acc={metrics.accuracy:.3f}  "
        f"sens={metrics.sensitivity:.3f}  spec={metrics.specificity:.3f}  "
        f"f1={metrics.f1:.3f}  auc={metrics.roc_auc:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 0.93))
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, metrics, y_true, y_prob


# ---------------------------------------------------------------------------
# Error analysis — mining misclassifications
# ---------------------------------------------------------------------------

def find_misclassifications(model: tf.keras.Model,
                            dataset: tf.data.Dataset,
                            n: int = 3,
                            most_confident: bool = True):
    """
    Return the n most interesting misclassified examples for error analysis:
    (images, y_true, y_pred, y_prob) with the highest prediction confidence
    against the true label — exactly the cases worth discussing in the report.
    """
    rows = []
    for x, y in dataset:
        p = model.predict(x, verbose=0).ravel()
        yt = y.numpy().ravel().astype(int)
        yp = (p >= 0.5).astype(int)
        for i in range(len(yt)):
            if yt[i] != yp[i]:
                conf = p[i] if yp[i] == 1 else 1 - p[i]
                rows.append((x[i].numpy(), yt[i], yp[i], float(p[i]), float(conf)))
    rows.sort(key=lambda r: r[-1], reverse=most_confident)
    imgs, trues, preds, probs = [], [], [], []
    for img, yt, yp, p, _ in rows[:n]:
        imgs.append(img); trues.append(yt); preds.append(yp); probs.append(p)
    return imgs, trues, preds, probs


def plot_misclassifications(imgs, trues, preds, probs,
                            figsize: tuple[float, float] = (12, 4)):
    """Show misclassified cells with true vs predicted labels for the report."""
    fig, axes = plt.subplots(1, max(1, len(imgs)), figsize=figsize)
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, yt, yp, p in zip(axes, imgs, trues, preds, probs):
        ax.imshow(img.astype("uint8"))
        ax.set_title(f"true={LABEL_NAMES_BY_VALUE[yt]}\npred={LABEL_NAMES_BY_VALUE[yp]} (p={p:.2f})",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("Misclassified examples — error analysis")
    fig.tight_layout()
    return fig

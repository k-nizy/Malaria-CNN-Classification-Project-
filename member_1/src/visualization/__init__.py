"""
src/visualization — Public API for training/validation plots.

Re-exports from common.eval to provide a clean import surface.
All implementation lives in common/eval.py (shared, tested, stable).

Provides the 4 required plots per model:
1. Training vs Validation Accuracy
2. Training vs Validation Loss
3. Confusion Matrix
4. ROC Curve with AUC
Plus: Final evaluation figure (single-row 4-plot layout)
"""

from common.eval import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    final_evaluation_figure,
)

__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "final_evaluation_figure",
]
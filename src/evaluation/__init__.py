"""
src/evaluation — Public API for model evaluation metrics.

Re-exports from common.eval to provide a clean import surface.
All implementation lives in common/eval.py (shared, tested, stable).

Label convention (from common.data): 1 = Parasitized (positive/infected),
0 = Uninfected. Recall == Sensitivity; Specificity == TNR on class 0.
"""

from common.eval import (
    Metrics,
    collect_predictions,
    predictions_from_probs,
    evaluate_metrics,
    evaluate_model,
    find_misclassifications,
    plot_misclassifications,
    CLASS_NAMES,
    LABEL_NAMES_BY_VALUE,
)

__all__ = [
    "Metrics",
    "collect_predictions",
    "predictions_from_probs",
    "evaluate_metrics",
    "evaluate_model",
    "find_misclassifications",
    "plot_misclassifications",
    "CLASS_NAMES",
    "LABEL_NAMES_BY_VALUE",
]
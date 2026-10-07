"""
src — Public API package for the Malaria CNN project.

Provides clean imports for all shared utilities.
Implementation lives in common/ (tested, stable).
"""

from src.data import *
from src.preprocessing import *
from src.evaluation import *
from src.visualization import *
from src.explainability import *
from src.utils import *

__all__ = [
    # data
    "SEED", "IMG_SIZE", "BATCH_SIZE", "LABEL_MAP", "CLASS_NAMES",
    "DATA_URL", "ZIP_NAME", "EXPECTED_TOTAL", "EXPECTED_PER_CLASS",
    "DataDirs", "is_colab", "resolve_dirs", "download_cell_images",
    "extract_cell_images", "count_images", "get_data",
    "make_synthetic_cell_dataset", "build_datasets", "collect_labels",
    "split_fingerprint", "_flip_label",
    # evaluation
    "Metrics", "collect_predictions", "predictions_from_probs",
    "evaluate_metrics", "evaluate_model", "find_misclassifications",
    "plot_misclassifications", "CLASS_NAMES", "LABEL_NAMES_BY_VALUE",
    # visualization
    "plot_training_curves", "plot_confusion_matrix",
    "plot_roc_curve", "final_evaluation_figure",
    # explainability
    "make_gradcam_heatmap", "overlay_heatmap",
    "pick_gradcam_examples", "gradcam_report_grid",
    # utils
    "MODELS", "EXPERIMENT_NAMING_CONVENTION",
    "POSITIVE_CLASS", "NEGATIVE_CLASS", "CLASS_LABELS",
]
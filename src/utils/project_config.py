"""
src/utils/project_config.py — Project-wide configuration.

Single source of truth for model assignments, experiment naming, and class labels.
Update this file when team decisions are made.
"""

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single model in the project."""
    name: str
    type: Literal["custom", "transfer_learning"]
    owner: str  # "Member 1", "Member 2", etc.
    architecture: str | None = None  # For transfer: backbone name (e.g., "MobileNetV2")
    notes: str = ""


# Dictionary of all 4 models — update owners when finalized
MODELS: dict[str, ModelConfig] = {
    "custom_resnet": ModelConfig(
        name="Custom ResNet",
        type="custom",
        owner="Member 2",  # Current assignment
        architecture="CustomResNet (Subclassing API)",
        notes="From-scratch ResNet with residual blocks, configurable width/depth"
    ),
    "transfer_model_1": ModelConfig(
        name="Transfer Model 1",
        type="transfer_learning",
        owner="Member 1",  # Group leader's own model
        architecture="MobileNetV3Small",  # Team decision — Member 1 workstream
        notes="Pretrained backbone, frozen baseline → progressive unfreezing"
    ),
    "transfer_model_2": ModelConfig(
        name="Transfer Model 2",
        type="transfer_learning",
        owner="TBD",
        architecture="TBD",  # Must be DIFFERENT from transfer_model_1
        notes="Different architecture from Transfer Model 1 for meaningful comparison"
    ),
    "custom_cnn": ModelConfig(
        name="Additional Custom CNN",
        type="custom",
        owner="Member 4",  # Current assignment
        architecture="TBD",  # Different from Custom ResNet
        notes="From-scratch, Subclassing API, distinct architecture from Custom ResNet"
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a specific model by key."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    return MODELS[model_key]


def get_model_owner(model_key: str) -> str:
    """Get the owner (member) assigned to a model."""
    return get_model_config(model_key).owner


def get_transfer_model_architectures() -> tuple[str, str]:
    """Get the two transfer model architectures (for comparison)."""
    return (
        MODELS["transfer_model_1"].architecture,
        MODELS["transfer_model_2"].architecture,
    )


# ---------------------------------------------------------------------------
# Experiment Naming Convention
# ---------------------------------------------------------------------------

EXPERIMENT_NAMING_CONVENTION = (
    "{model_key}_exp_{exp_id:02d}_{description}"
    .lower()
    .replace(" ", "_")
)

# Examples:
# custom_resnet_exp_01_baseline
# custom_resnet_exp_02_lr_1e-4
# transfer_model_1_exp_01_baseline
# transfer_model_1_exp_02_unfreeze_last20
# transfer_model_2_exp_01_baseline
# custom_cnn_exp_01_baseline


def make_experiment_name(model_key: str, exp_id: int, description: str) -> str:
    """Generate a standardized experiment run name."""
    return EXPERIMENT_NAMING_CONVENTION.format(
        model_key=model_key,
        exp_id=exp_id,
        description=description.lower().replace(" ", "_")
    )


# ---------------------------------------------------------------------------
# Class Labels (Critical for Metrics)
# ---------------------------------------------------------------------------

# DO NOT CHANGE — this defines the positive/infected class
POSITIVE_CLASS: int = 1          # Parasitized = infected
NEGATIVE_CLASS: int = 0          # Uninfected

CLASS_LABELS: dict[int, str] = {
    POSITIVE_CLASS: "Parasitized",
    NEGATIVE_CLASS: "Uninfected",
}

# Sensitivity = Recall on positive class (Parasitized)
# Specificity = TNR on negative class (Uninfected)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]  # malaria-cnn-formative-2/

DEFAULT_LOGDIR = PROJECT_ROOT / "logs" / "tensorboard"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "report"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_tensorboard_dir(model_key: str) -> Path:
    """Get the TensorBoard log directory for a model."""
    return DEFAULT_LOGDIR / model_key


def get_results_dir(subdir: str = "") -> Path:
    """Get a results subdirectory."""
    return DEFAULT_RESULTS_DIR / subdir


# ---------------------------------------------------------------------------
# Training Defaults (shared across all models)
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LR = 1e-3
DEFAULT_OPTIMIZER = "adam"

# Transfer learning specific defaults
TRANSFER_DEFAULTS = {
    "freeze_backbone": True,
    "dropout": 0.3,
    "dense_units": 256,
    "l2": 0.0,
}

# Custom model defaults
CUSTOM_DEFAULTS = {
    "dropout": 0.3,
    "l2": 0.0,
    "width_mult": 1.0,
    "blocks_per_stage": 2,
    "bn_momentum": 0.99,
}


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_model_assignments() -> list[str]:
    """Check that all models have owners assigned. Returns list of warnings."""
    warnings = []
    for key, config in MODELS.items():
        if config.owner == "TBD":
            warnings.append(f"Model '{key}' ({config.name}) has no owner assigned")
        if config.type == "transfer_learning" and config.architecture == "TBD":
            warnings.append(f"Transfer model '{key}' has no architecture selected")
    if MODELS["transfer_model_1"].architecture == MODELS["transfer_model_2"].architecture:
        if MODELS["transfer_model_1"].architecture != "TBD":
            warnings.append("Transfer Model 1 and 2 have the SAME architecture — must be different for comparison")
    return warnings


# Run validation on import (prints warnings but doesn't raise)
_warnings = validate_model_assignments()
if _warnings:
    import warnings as pywarnings
    for w in _warnings:
        pywarnings.warn(f"[project_config] {w}", UserWarning, stacklevel=2)
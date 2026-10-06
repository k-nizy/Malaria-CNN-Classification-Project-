"""
src/data — Public API for data pipeline.

Re-exports from common.data to provide a clean import surface.
All implementation lives in common/data.py (shared, tested, stable).
"""

from common.data import (
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
    LABEL_MAP,
    CLASS_NAMES,
    DATA_URL,
    ZIP_NAME,
    EXPECTED_TOTAL,
    EXPECTED_PER_CLASS,
    DataDirs,
    is_colab,
    resolve_dirs,
    download_cell_images,
    extract_cell_images,
    count_images,
    get_data,
    make_synthetic_cell_dataset,
    build_datasets,
    collect_labels,
    split_fingerprint,
    _flip_label,
)

__all__ = [
    "SEED",
    "IMG_SIZE",
    "BATCH_SIZE",
    "LABEL_MAP",
    "CLASS_NAMES",
    "DATA_URL",
    "ZIP_NAME",
    "EXPECTED_TOTAL",
    "EXPECTED_PER_CLASS",
    "DataDirs",
    "is_colab",
    "resolve_dirs",
    "download_cell_images",
    "extract_cell_images",
    "count_images",
    "get_data",
    "make_synthetic_cell_dataset",
    "build_datasets",
    "collect_labels",
    "split_fingerprint",
    "_flip_label",
]
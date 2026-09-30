"""
src/preprocessing — Public API for data augmentation.

Re-exports augmentation utilities from common.data.
All implementation lives in common/data.py (shared, tested, stable).
"""

from common.data import build_datasets

__all__ = ["build_datasets"]
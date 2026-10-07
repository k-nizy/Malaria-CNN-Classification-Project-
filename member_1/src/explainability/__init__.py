"""
src/explainability — Public API for Grad-CAM explainability.

Re-exports from common.gradcam to provide a clean import surface.
All implementation lives in common/gradcam.py (shared, tested, stable).

Works for both:
- Subclassed models (CustomResNet) — uses self._last_conv_features tap
- Functional/Sequential models — auto-installs marker on last Conv2D layer
"""

from common.gradcam import (
    make_gradcam_heatmap,
    overlay_heatmap,
    pick_gradcam_examples,
    gradcam_report_grid,
)

__all__ = [
    "make_gradcam_heatmap",
    "overlay_heatmap",
    "pick_gradcam_examples",
    "gradcam_report_grid",
]
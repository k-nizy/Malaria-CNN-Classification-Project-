"""
common/gradcam.py — Grad-CAM for both functional and subclassed Keras models.

Assignment §5.4 trap: "Fully encapsulated models that hide convolutional
outputs behind nested blocks will fail to generate Grad-CAM heatmaps."

This module solves that with an EAGER implementation: instead of building an
intermediate sub-model (which needs layer traversal and breaks on subclassed
models), we run a forward pass under GradientTape, grabbing the last
convolutional feature map from inside the model's own call() via a marker
attribute (self._last_conv_features). Works for:
  - our subclassed CustomResNet (marker set inside call()),
  - any functional/Sequential CNN (marker auto-installed around the LAST
    Conv2D layer found by depth-first search).

Reference: Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization."
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Marker plumbing — how we tap the conv features without sub-models
# ---------------------------------------------------------------------------

def _install_marker(model: tf.keras.Model) -> None:
    """
    Ensure model.call() records its last conv feature map on
    `model._last_conv_features`.

    Subclassed models (our CustomResNet) set the marker themselves inside
    call(). For Sequential/functional models we monkey-patch the LAST Conv2D
    layer's call to stash its output on the model.
    """
    if getattr(model, "_gradcam_marker_ready", False):
        return

    if hasattr(model, "_last_conv_features"):        # subclassed: already taps
        model._gradcam_marker_ready = True
        return

    last_conv = _find_last_conv_layer(model)
    if last_conv is None:
        raise ValueError(
            f"No Conv2D layer found on {type(model).__name__}; Grad-CAM needs one.")

    original_call = last_conv.call

    def call_and_mark(inputs, *args, **kwargs):
        out = original_call(inputs, *args, **kwargs)
        model._last_conv_features = out
        return out

    last_conv.call = call_and_mark
    model._gradcam_marker_ready = True
    model._gradcam_patched_layer = last_conv


def _find_last_conv_layer(model: tf.keras.Model):
    """Depth-first search for the last (in build order) Conv2D layer."""
    convs = []

    def walk(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            convs.append(layer)
        for sub in getattr(layer, "layers", []):      # Sequential/nested
            walk(sub)

    walk(model)
    return convs[-1] if convs else None


# ---------------------------------------------------------------------------
# Heatmap computation
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(img_array: np.ndarray | tf.Tensor,
                         model: tf.keras.Model,
                         pred_index: int | None = None) -> tuple[np.ndarray, float]:
    """
    Compute a Grad-CAM heatmap for one image.

    Args:
        img_array: HxWx3 float32 array (0-255 scale, unnormalized —
            normalization happens inside the model, matching training).
        model: trained Keras model (Sequential, functional, or subclassed).
        pred_index: class to explain (1=parasitized, 0=uninfected).
            None = explain the predicted class.

    Returns:
        (heatmap, prob) — heatmap is HxW normalized to [0, 1] at the conv
        feature resolution; prob is the model's P(parasitized).

    Implementation notes (two paths):
      - Subclassed models (our CustomResNet): the conv trunk runs OUTSIDE the
        tape, then the head is re-computed INSIDE the tape with the conv
        feature map explicitly watched. This is the unambiguous way to get
        d(score)/d(conv) for subclassed models (assignment §5.4).
      - Functional/Sequential models: a default tape (variables auto-watched)
        re-runs the model so the conv output is an intermediate of the
        recorded graph — the canonical keras.io Grad-CAM pattern.
    """
    _install_marker(model)

    x = tf.convert_to_tensor(np.asarray(img_array, dtype=np.float32))
    if x.shape.rank == 3:
        x = x[tf.newaxis, ...]                       # add batch dim

    # ---- forward pass to obtain conv features + probability --------------
    model._last_conv_features = None
    prob_tensor = model(x, training=False)
    conv = model._last_conv_features
    if conv is None:
        raise RuntimeError(
            "Model did not record _last_conv_features — Grad-CAM tap failed.")

    prob = float(tf.squeeze(prob_tensor).numpy())
    if pred_index is None:
        pred_index = 1 if prob >= 0.5 else 0

    is_subclassed = all(hasattr(model, part) for part in ("stem", "gap", "head"))

    if is_subclassed:
        # head recompute with watched conv — guaranteed gradient path.
        # We use the LOGIT (pre-sigmoid z = yW + b), NOT the probability:
        # a well-trained sigmoid saturates at p≈1.0 and σ'(z)=σ(1-σ)≈0
        # kills the gradient (verified: perfect model produced all-zero
        # heatmaps via the probability path). Grad-CAM on logits is the
        # standard remedy; the logit is monotone in p so localization is
        # identical. Explaining class 0 = negating the logit.
        with tf.GradientTape() as tape:
            tape.watch(conv)
            y = model.gap(conv)
            y = model.dropout(y, training=False)
            z = tf.matmul(y, model.head.kernel) + model.head.bias   # logit
            score = tf.squeeze(z, axis=-1)                          # (1,)
            if pred_index == 0:                                     # uninfected
                score = -score
        grads = tape.gradient(score, conv)
    else:
        # functional/Sequential: re-run under a default (variable-watching)
        # tape; conv output is then an intermediate of the recorded graph.
        # Score on the probability — typical classifiers are not saturated;
        # if a functional model ever saturates, prefer the subclassed-style
        # logit approach (split the final Dense from its activation).
        with tf.GradientTape() as tape:
            model._last_conv_features = None
            p2 = model(x, training=False)
            conv = model._last_conv_features
            score = tf.squeeze(p2, axis=-1)
            if pred_index == 0:
                score = 1.0 - score
        grads = tape.gradient(score, conv)

    if grads is None:
        raise RuntimeError("Gradient wrt conv features is None — see "
                           "common/gradcam.py docstring for the two paths.")

    # channel-wise mean gradient = importance weights (alpha_k in the paper)
    weights = tf.reduce_mean(grads, axis=(0, 1))     # (C,)
    # weighted sum of activation maps, ReLU'd (only positive evidence)
    heatmap = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))  # (B,h,w)
    heatmap = heatmap[0]
    mx = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap.numpy(), prob


def overlay_heatmap(img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4,
                    ax=None):
    """Overlay a Jet heatmap on the (uint8) cell image. Draws on `ax` if given."""
    hm = tf.image.resize(heatmap[..., np.newaxis],
                         (img.shape[0], img.shape[1]),
                         method="bilinear").numpy()[..., 0]
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    colored = cm.jet(hm)[..., :3]                    # HxWx3 in [0,1]
    overlay = alpha * colored + (1 - alpha) * (img / 255.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Example picking + report grid
# ---------------------------------------------------------------------------

def pick_gradcam_examples(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          images: np.ndarray):
    """
    Pick the assignment's required example types from a test stream:
      (best correct-infected, best correct-uninfected, first incorrect)
    Returns list of (image, y_true, y_pred, prob).
    """
    y_pred = (y_prob >= 0.5).astype(int)
    correct = y_pred == y_true
    wrong = ~correct

    idx_ci = np.where(correct & (y_true == 1))[0]
    idx_cu = np.where(correct & (y_true == 0))[0]

    # most confident correct examples
    conf = np.where(y_pred == 1, y_prob, 1 - y_prob)
    ex_inf = idx_ci[np.argmax(conf[idx_ci])] if len(idx_ci) else None
    ex_uninf = idx_cu[np.argmax(conf[idx_cu])] if len(idx_cu) else None
    ex_wrong = np.where(wrong)[0][0] if wrong.any() else None

    out = []
    for idx in (ex_inf, ex_uninf, ex_wrong):
        if idx is not None:
            out.append((images[idx], int(y_true[idx]), int(y_pred[idx]),
                        float(y_prob[idx])))
    return out


def gradcam_report_grid(examples, model,
                        save_path: str | None = None,
                        figsize=(12, 4.5)):
    """
    The report figure: one row of (original, heatmap-overlay) per example with
    true/pred labels — the 3 assignment-required Grad-CAM cases per model.
    """
    fig, axes = plt.subplots(2, len(examples), figsize=figsize)
    if len(examples) == 1:
        axes = axes.reshape(2, 1)

    for col, (img, yt, yp, prob) in enumerate(examples):
        heat, p = make_gradcam_heatmap(img, model)

        axes[0, col].imshow(img.astype("uint8"))
        axes[0, col].set_title(
            f"true={'Parasitized' if yt else 'Uninfected'}\n"
            f"pred={'Parasitized' if yp else 'Uninfected'} (p={p:.2f})",
            fontsize=9)
        axes[0, col].axis("off")

        overlay_heatmap(img, heat, ax=axes[1, col])
        axes[1, col].set_ylabel("Grad-CAM")

    axes[0, 0].text(-0.25, 0.5, "input", rotation=90, va="center",
                    transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.25, 0.5, "Grad-CAM", rotation=90, va="center",
                    transform=axes[1, 0].transAxes)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

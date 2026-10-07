"""
Colombe's DenseNet121 — transfer learning for malaria cell images.

This is Member 3's model. It is not the group's custom ResNet and it is not
Peggy's MobileNetV2. DenseNet121 is loaded from ImageNet (Huang et al., 2017)
and a new head is trained on the NIH malaria set. Later runs unfreeze the
top of the backbone.

Design choices that the oral defense will ask about
----------------------------------------------------
* Preprocessing sits inside the model. DenseNet's preprocess_input expects
  raw 0-255 RGB and does the Caffe-style BGR mean subtraction itself.
  The shared tf.data pipeline already yields 0-255, so a second /255 would
  break the pretrained filters.
* The head is GlobalAveragePooling -> Dropout -> Dense(logits) -> sigmoid.
  The logit is a separate layer so Grad-CAM can differentiate it. A saturated
  sigmoid has a near-zero derivative and the heatmap goes blank.
* Batch-norm layers stay frozen even when convolutional layers are unfrozen.
  Batch 16 is too small for stable batch statistics; ImageNet moving averages
  are the safer choice.
* The winner used for the four report plots is chosen by validation loss,
  not by test-set score. Test metrics are still computed for every run
  because the assignment table requires them.

The split is the group's shared one: common.data.build_datasets, seed 42,
70% train / 15% val / 15% test. Label 1 = Parasitized (infected).
"""

from __future__ import annotations

import csv
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from common.data import SEED, build_datasets
from common.eval import (
    LABEL_NAMES_BY_VALUE,
    evaluate_metrics,
    evaluate_model,
    final_evaluation_figure,
    plot_misclassifications,
)
from common.tb import ExperimentLogger, make_run_name

MODEL_NAME = "densenet121"


def evaluation_datasets(spec: dict, cell_images):
    """
    Clean train/val/test at this run's image size.

    Augmentation is off. The file split still uses seed 42 and a 30% holdout
    cut in half, so the test files match the training run.
    """
    return build_datasets(
        cell_images,
        img_size=tuple(spec["img_size"]),
        batch_size=int(spec["batch_size"]),
        validation_fraction=0.30,
        seed=SEED,
        augment=False,
    )


def default_experiments() -> list[dict]:
    """
    Seven runs. Each run changes one decision from the run it is compared with.

    01 is the reference (frozen, 128px).
    02 changes only the input size.
    03 adds augmentation.
    04 lowers the head learning rate.
    05 starts fine-tuning, and only the last dense block (conv5), at 1e-5.
    06 is the depth branch: unfreeze from conv4 as well, same learning rate.
    07 goes back to the conv5 depth and only raises dropout 0.3 -> 0.4.

    Batch size stays 16 on every run. That is the size that fits a 224px
    DenseNet121 on a free Colab T4, and holding it fixed means batch size
    is not a hidden variable.
    """
    shared = dict(batch_size=16, optimizer="adam")
    return [
        dict(
            id=1, name="baseline_frozen_128",
            img_size=(128, 128), lr=1e-3, dropout=0.3, l2=0.0,
            augment=False, finetune="frozen",
            note="Frozen ImageNet trunk, new head only, 128px. Reference.",
            **shared,
        ),
        dict(
            id=2, name="input_224",
            img_size=(224, 224), lr=1e-3, dropout=0.3, l2=0.0,
            augment=False, finetune="frozen",
            note="Same frozen head, but 224px — the resolution DenseNet was pretrained at.",
            **shared,
        ),
        dict(
            id=3, name="augment",
            img_size=(224, 224), lr=1e-3, dropout=0.3, l2=0.0,
            augment=True, finetune="frozen",
            note="Add flips, small rotations, and zoom. Cells have no up/down.",
            **shared,
        ),
        dict(
            id=4, name="head_lr_1e-4",
            img_size=(224, 224), lr=1e-4, dropout=0.3, l2=0.0,
            augment=True, finetune="frozen",
            note="Lower the head learning rate before any backbone weights move.",
            **shared,
        ),
        dict(
            id=5, name="unfreeze_conv5",
            img_size=(224, 224), lr=1e-5, dropout=0.3, l2=0.0,
            augment=True, finetune="conv5",
            note="Unfreeze the last dense block only. Learning rate drops to 1e-5. BN stays frozen.",
            **shared,
        ),
        dict(
            id=6, name="unfreeze_from_conv4",
            img_size=(224, 224), lr=1e-5, dropout=0.3, l2=0.0,
            augment=True, finetune="conv4",
            note="Deeper fine-tune: conv4 and conv5. Same 1e-5. This is the run most likely to overfit.",
            **shared,
        ),
        dict(
            id=7, name="final_conv5_dropout04",
            img_size=(224, 224), lr=1e-5, dropout=0.4, l2=0.0,
            augment=True, finetune="conv5",
            note="Back to conv5 depth, dropout 0.3 to 0.4. One change from exp 05.",
            **shared,
        ),
    ]


def _finetune_strategy(finetune: str) -> str:
    if finetune in (None, "frozen", "none"):
        return "frozen_head"
    if finetune == "conv5":
        return "finetune_conv5_bn_frozen"
    if finetune == "conv4":
        return "finetune_from_conv4_bn_frozen"
    return f"finetune_{finetune}_bn_frozen"


def _apply_finetune(base: keras.Model, finetune: str | int) -> tuple[int, str]:
    """
    Freeze or unfreeze the backbone.

    Returns (n_tail_layers, name_of_first_trainable_candidate).
    Every BatchNormalization layer is forced off afterwards, including
    inside the unfrozen tail. Small batches make BN statistics noisy.
    """
    if finetune in (0, None, "frozen", "none"):
        base.trainable = False
        return 0, "none"

    if isinstance(finetune, int):
        n = finetune
    else:
        prefix = str(finetune)
        start = next(
            (i for i, layer in enumerate(base.layers) if layer.name.startswith(prefix)),
            None,
        )
        if start is None:
            raise ValueError(f"No DenseNet layer starts with {prefix!r}")
        n = len(base.layers) - start

    # Parent flag first. If the parent stays trainable=False, child flags
    # are ignored and the "unfreeze" run secretly trains nothing.
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers[-n:]:
        layer.trainable = True
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    first = base.layers[-n].name
    return n, first


def build_densenet121(
    img_size: tuple[int, int] = (224, 224),
    dropout: float = 0.3,
    l2: float = 0.0,
    finetune: str | int = "frozen",
    weights: str | None = "imagenet",
) -> keras.Model:
    """
    ImageNet DenseNet121 + a small malaria head.

    `model.grad_model` maps the raw cell image to (last feature map, logit)
    so Grad-CAM does not have to poke inside the nested backbone.
    """
    if weights == "none":
        weights = None

    inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="cell_rgb")
    x = layers.Lambda(
        lambda t: keras.applications.densenet.preprocess_input(t),
        name="densenet_preprocess",
    )(inputs)

    base = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(img_size[0], img_size[1], 3),
        name="densenet121",
    )
    n_unfrozen, first_layer = _apply_finetune(base, finetune)

    # training=False keeps dropout inside the backbone off. DenseNet121 has
    # none, but BN with trainable=False already runs in inference mode.
    features = base(x, training=False)
    pooled = layers.GlobalAveragePooling2D(name="gap")(features)
    dropped = layers.Dropout(dropout, name="head_dropout")(pooled)
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    logits = layers.Dense(1, kernel_regularizer=reg, name="logits")(dropped)
    probs = layers.Activation("sigmoid", name="classifier")(logits)

    model = keras.Model(inputs, probs, name="densenet121_malaria")
    model.grad_model = keras.Model(
        inputs, [features, logits], name="densenet121_gradcam"
    )
    model.backbone = base
    model.finetune_info = {
        "n_tail_layers": n_unfrozen,
        "first_tail_layer": first_layer,
        "strategy": _finetune_strategy(finetune if not isinstance(finetune, int) else f"last_{finetune}"),
    }
    return model


def count_params(weights) -> int:
    return int(sum(int(tf.size(w).numpy()) for w in weights))


def describe_model(model: keras.Model) -> str:
    info = model.finetune_info
    n_total = count_params(model.weights)
    n_train = count_params(model.trainable_weights)
    bn_on = [
        layer.name
        for layer in model.backbone.layers
        if isinstance(layer, layers.BatchNormalization) and layer.trainable
    ]
    return (
        f"params total={n_total:,}  trainable={n_train:,}  "
        f"tail_layers={info['n_tail_layers']}  "
        f"first_tail_layer={info['first_tail_layer']}  "
        f"strategy={info['strategy']}  "
        f"trainable_bn={len(bn_on)}"
    )


def _history_dict(history: keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def _best_epoch_stats(history: dict) -> dict[str, float]:
    val_loss = history["val_loss"]
    best = int(np.argmin(val_loss))
    return {
        "best_epoch": float(best + 1),  # 1-based, for the report
        "best_val_loss": float(val_loss[best]),
        "best_val_accuracy": float(history["val_accuracy"][best]),
        "final_train_accuracy": float(history["accuracy"][-1]),
        "final_val_accuracy": float(history["val_accuracy"][-1]),
        "epochs_ran": float(len(val_loss)),
    }


def _read_metrics_file(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def _read_history_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"empty training log: {path}")
    keys = [k for k in rows[0] if k != "epoch"]
    history = {k: [] for k in keys}
    for row in rows:
        for key in keys:
            history[key].append(float(row[key]))
    return history


class _History:
    """Small stand-in so common.eval.final_evaluation_figure can plot curves."""

    def __init__(self, history: dict):
        self.history = history


def _require_tensorboard() -> None:
    """The assignment logs every run to TensorBoard. Fail early if it is missing."""
    try:
        import tensorboard  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is not installed, so the training logs cannot be written. "
            "Install it with: pip install tensorboard"
        ) from exc


def run_experiment(
    spec: dict,
    cell_images,
    epochs: int = 10,
    logdir: str | Path = "logs/fit",
    models_dir: str | Path = "models/densenet121",
    patience: int = 3,
    verbose: int = 1,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    weights: str | None = "imagenet",
    force: bool = False,
) -> dict:
    """
    Train one experiment, or reload it if weights and test metrics already exist.

    Test metrics are written into the same TensorBoard run as the curves.
    """
    _require_tensorboard()
    run_name = make_run_name(MODEL_NAME, spec["id"], spec["name"])
    log_dir = Path(logdir) / run_name
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"{run_name}.weights.h5"
    spec_path = models_dir / f"{run_name}.spec.json"
    metrics_path = log_dir / "final_metrics.txt"
    history_path = log_dir / "training_log.csv"

    serializable = {
        **spec,
        "img_size": list(spec["img_size"]),
        "weights": weights,
    }
    if (
        not force
        and metrics_path.exists()
        and weights_path.exists()
        and history_path.exists()
    ):
        print(f"[{run_name}] already finished — loading saved metrics")
        history = _read_history_csv(history_path)
        metrics = _read_metrics_file(metrics_path)
        return {
            "run": run_name,
            "spec": spec,
            "history": history,
            "metrics": metrics,
            "weights_path": str(weights_path),
            "log_dir": str(log_dir),
            "skipped": True,
        }

    if log_dir.exists() and not metrics_path.exists():
        shutil.rmtree(log_dir)

    tf.keras.utils.set_random_seed(SEED)
    img_size = tuple(spec["img_size"])
    train_ds, val_ds, test_ds = build_datasets(
        cell_images,
        img_size=img_size,
        batch_size=spec["batch_size"],
        validation_fraction=0.30,
        seed=SEED,
        augment=bool(spec["augment"]),
    )

    model = build_densenet121(
        img_size=img_size,
        dropout=spec["dropout"],
        l2=spec["l2"],
        finetune=spec["finetune"],
        weights=weights,
    )
    print(f"[{run_name}] {describe_model(model)}")
    print(f"[{run_name}] {spec['note']}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=spec["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    strategy = model.finetune_info["strategy"]
    exp = ExperimentLogger(
        model_name=MODEL_NAME,
        exp_id=spec["id"],
        description=spec["name"],
        hparams=dict(
            lr=float(spec["lr"]),
            optimizer=spec.get("optimizer", "adam"),
            batch_size=int(spec["batch_size"]),
            epochs=int(epochs),
            augmentation=bool(spec["augment"]),
            dropout=float(spec["dropout"]),
            l2=float(spec["l2"]),
            img_size=int(img_size[0]),
            unfreeze_last=int(model.finetune_info["n_tail_layers"]),
            strategy=strategy,
            model="DenseNet121",
            first_tail_layer=model.finetune_info["first_tail_layer"],
        ),
        logdir=logdir,
    )

    fit_kwargs = dict(
        validation_data=val_ds,
        epochs=epochs,
        callbacks=exp.callbacks(early_stopping_patience=patience),
        verbose=verbose,
    )
    if steps_per_epoch:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch
    if validation_steps:
        fit_kwargs["validation_steps"] = validation_steps

    history_obj = model.fit(train_ds, **fit_kwargs)
    history = _history_dict(history_obj)

    # EarlyStopping(restore_best_weights=True) has put the best val-loss
    # weights back. Save those, not a later overfit epoch.
    model.save_weights(weights_path)
    spec_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    metrics, _, _ = evaluate_model(model, test_ds)
    row = metrics.as_row()
    row.update(_best_epoch_stats(history))
    exp.log_final_metrics(row)
    exp.log_hparams_once(row)
    print(f"[{run_name}] {metrics.format()}")

    result = {
        "run": run_name,
        "spec": spec,
        "history": history,
        "metrics": {k: float(v) for k, v in row.items()},
        "weights_path": str(weights_path),
        "log_dir": str(exp.log_dir),
        "skipped": False,
    }
    # Drop the graph before the next experiment. DenseNet121 at 224 is large.
    del model, train_ds, val_ds, test_ds
    keras.backend.clear_session()
    gc.collect()
    return result


def run_campaign(cell_images, experiments=None, **kwargs) -> list[dict]:
    experiments = experiments or default_experiments()
    results = []
    for spec in experiments:
        results.append(run_experiment(spec, cell_images, **kwargs))
    return results


def best_by_val_loss(results: list[dict]) -> dict:
    """
    Pick the run whose best validation loss is lowest.

    Test sensitivity is not used here. The test set is for the table and
    the final plots, not for shopping among the seven runs.
    """
    def key(result: dict):
        stats = _best_epoch_stats(result["history"])
        return (stats["best_val_loss"], -stats["best_val_accuracy"])

    return min(results, key=key)


def load_trained(result: dict, weights: str | None = None) -> keras.Model:
    """Rebuild the architecture and load the saved weights."""
    spec = result["spec"]
    # Initial weights do not matter: load_weights replaces them, including
    # the backbone and the BatchNorm moving statistics.
    model = build_densenet121(
        img_size=tuple(spec["img_size"]),
        dropout=spec["dropout"],
        l2=spec["l2"],
        finetune=spec["finetune"],
        weights=weights,
    )
    model.load_weights(result["weights_path"])
    return model


def threshold_rows(y_true: np.ndarray, y_prob: np.ndarray,
                   thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)) -> list[dict]:
    """Same probabilities, several decision thresholds. For the ROC discussion."""
    rows = []
    for threshold in thresholds:
        metrics = evaluate_metrics(y_true, y_prob, threshold=threshold)
        row = {"threshold": threshold}
        row.update(metrics.as_row())
        rows.append(row)
    return rows


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def gradcam_heatmap(model: keras.Model, image: np.ndarray,
                    class_index: int | None = None) -> tuple[np.ndarray, float]:
    """
    Grad-CAM on the backbone's final feature map, using the pre-sigmoid logit.

    class_index 1 explains "parasitized", 0 explains "uninfected" (the logit
    is negated). None explains whichever class the model predicted.
    """
    x = tf.convert_to_tensor(np.asarray(image, dtype=np.float32))
    if x.shape.rank == 3:
        x = x[tf.newaxis, ...]

    with tf.GradientTape() as tape:
        features, logits = model.grad_model(x, training=False)
        logit = tf.squeeze(logits, axis=-1)
        prob = float(tf.sigmoid(logit)[0].numpy())
        if class_index is None:
            class_index = 1 if prob >= 0.5 else 0
        score = logit if class_index == 1 else -logit
    grads = tape.gradient(score, features)
    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradient is None. The feature map is not connected to the logit."
        )

    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights[:, tf.newaxis, tf.newaxis, :], axis=-1))
    heatmap = heatmap[0]
    peak = tf.reduce_max(heatmap)
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap.numpy(), prob


def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    hm = tf.image.resize(
        heatmap[..., np.newaxis],
        (image.shape[0], image.shape[1]),
        method="bilinear",
    ).numpy()[..., 0]
    colored = cm.jet(np.clip(hm, 0, 1))[..., :3]
    return np.clip(alpha * colored + (1 - alpha) * (image / 255.0), 0, 1)


def collect_case_studies(model: keras.Model, dataset: tf.data.Dataset,
                         n_errors: int = 3) -> dict:
    """
    One pass over the test set.

    Keeps the most confident correct infected cell, the most confident correct
    uninfected cell, the first-needed incorrect case (highest confidence),
    and the n_errors most confident mistakes for the error analysis.
    """
    best_infected = None
    best_uninfected = None
    mistakes = []

    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0).ravel()
        images = batch_x.numpy()
        labels = batch_y.numpy().ravel().astype(int)
        preds = (probs >= 0.5).astype(int)
        for image, label, pred, prob in zip(images, labels, preds, probs):
            confidence = float(prob if pred == 1 else 1.0 - prob)
            record = (confidence, image, int(label), int(pred), float(prob))
            if label == pred and label == 1:
                if best_infected is None or confidence > best_infected[0]:
                    best_infected = record
            elif label == pred and label == 0:
                if best_uninfected is None or confidence > best_uninfected[0]:
                    best_uninfected = record
            elif label != pred:
                mistakes.append(record)

    mistakes.sort(key=lambda row: row[0], reverse=True)
    return {
        "correct_infected": best_infected,
        "correct_uninfected": best_uninfected,
        "incorrect": mistakes[0] if mistakes else None,
        "mistakes": mistakes[:n_errors],
    }


def save_gradcam_figure(model, cases: dict, save_path: str | Path):
    import matplotlib.pyplot as plt

    panels = [
        ("Correct infected", cases.get("correct_infected")),
        ("Correct uninfected", cases.get("correct_uninfected")),
        ("Incorrect", cases.get("incorrect")),
    ]
    panels = [(title, row) for title, row in panels if row is not None]
    if not panels:
        raise RuntimeError("No test images were available for Grad-CAM.")

    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 7.2))
    if len(panels) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, (title, row) in enumerate(panels):
        _, image, label, pred, prob = row
        heat, _ = gradcam_heatmap(model, image, class_index=pred)
        axes[0, col].imshow(image.astype("uint8"))
        axes[0, col].set_title(
            f"{title}\ntrue={LABEL_NAMES_BY_VALUE[label]}\n"
            f"pred={LABEL_NAMES_BY_VALUE[pred]}  p={prob:.2f}",
            fontsize=9,
        )
        axes[0, col].axis("off")
        axes[1, col].imshow(_overlay(image, heat))
        axes[1, col].set_title("Grad-CAM (predicted class)", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("DenseNet121 Grad-CAM — correct infected, correct uninfected, incorrect", fontsize=12)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def save_error_figure(cases: dict, save_path: str | Path):
    mistakes = cases["mistakes"]
    if not mistakes:
        return None
    imgs = [row[1] for row in mistakes]
    trues = [row[2] for row in mistakes]
    preds = [row[3] for row in mistakes]
    probs = [row[4] for row in mistakes]
    fig = plot_misclassifications(imgs, trues, preds, probs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def save_final_report_pack(model, result: dict, test_ds, out_dir: str | Path) -> dict:
    """Four required plots, threshold table, Grad-CAM, and misclassification grid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = _History(result["history"])
    fig, metrics, y_true, y_prob = final_evaluation_figure(
        model,
        history,
        test_ds,
        save_path=str(out_dir / "final_four_plots.png"),
    )
    counts = confusion_counts(y_true, y_prob, threshold=0.5)
    thresholds = threshold_rows(y_true, y_prob)
    cases = collect_case_studies(model, test_ds, n_errors=3)
    gradcam_fig = save_gradcam_figure(model, cases, out_dir / "gradcam.png")
    error_fig = save_error_figure(cases, out_dir / "misclassifications.png")

    table_path = out_dir / "threshold_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(thresholds[0].keys()))
        writer.writeheader()
        writer.writerows(thresholds)

    summary = {
        "run": result["run"],
        "metrics": metrics.as_row(),
        "counts_at_0.5": counts,
        "thresholds": thresholds,
        "best_epoch": _best_epoch_stats(result["history"]),
        "n_mistakes_shown": len(cases["mistakes"]),
        "has_incorrect_gradcam": cases["incorrect"] is not None,
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Confusion at 0.5: TP={counts['tp']}  FN={counts['fn']} (missed infections)  "
        f"TN={counts['tn']}  FP={counts['fp']}"
    )
    print(metrics.format())
    return {
        "fig": fig,
        "gradcam_fig": gradcam_fig,
        "error_fig": error_fig,
        "summary": summary,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def write_results_csv(results: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {
            "experiment": result["run"],
            "note": result["spec"]["note"],
            "img_size": result["spec"]["img_size"][0],
            "batch_size": result["spec"]["batch_size"],
            "lr": result["spec"]["lr"],
            "dropout": result["spec"]["dropout"],
            "l2": result["spec"]["l2"],
            "augment": result["spec"]["augment"],
            "finetune": result["spec"]["finetune"],
            "optimizer": result["spec"].get("optimizer", "adam"),
        }
        row.update(result["metrics"])
        rows.append(row)
    # Union of keys, stable order: config first, then whatever metrics appeared.
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

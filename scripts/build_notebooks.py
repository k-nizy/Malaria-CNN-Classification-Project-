"""
scripts/build_notebooks.py — generate the three member notebooks.

Why generated: the three notebooks must share IDENTICAL, tested plumbing
(data split, metric names, TB logging, Grad-CAM) while carrying each member's
own prose, model, and experiment campaign. Generating from one script keeps
the plumbing in lockstep with common/ (it is copied from the modules that
44 tests verify) and makes regeneration after an API change a one-liner.

Validation is built in: every generated notebook is checked with nbformat's
schema validator AND every code cell is compile()-checked, so a syntax error
can never reach Colab. Run tests/test_notebook_glue.py for the behavioral
verification of the same glue flow.

    python scripts/build_notebooks.py            # build + validate + write
"""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Shared code that every notebook executes (copied verbatim from common/, the
# modules verified by tests/test_*.py — do not hand-edit inside the string).
# ---------------------------------------------------------------------------

ONE_SCRIPT = '''
# ---------------------------------------------------------------------------
# ONE-TIME SETUP — run this cell first in a fresh Colab session.
# It pulls the group's verified helper code from GitHub and prepares the
# environment. Everything later imports from `common`.
# ---------------------------------------------------------------------------
import os, sys, subprocess, pathlib

pathlib.Path("figures").mkdir(exist_ok=True)   # report figures land here

REPO_URL = "https://github.com/k-nizy/Malaria-CNN-Classification-Project-.git"
if not pathlib.Path("common").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    for item in ["common", "tests"]:
        src = pathlib.Path("Malaria-CNN-Classification-Project-") / item
        if src.exists():
            src.rename(item)
if str(pathlib.Path.cwd()) not in sys.path:
    sys.path.insert(0, str(pathlib.Path.cwd()))

# --- Google Drive: persist dataset + TensorBoard logs across Colab resets ---
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    os.environ["TB_LOGDIR"] = "/content/drive/MyDrive/malaria_project/logs/fit"
    DRIVE_DATA = "/content/drive/MyDrive/malaria_project/data"
else:  # local machine / pytest — everything under ./
    os.environ.setdefault("TB_LOGDIR", "logs/fit")
    DRIVE_DATA = "data"

import tensorflow as tf
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
'''

SETUP_CELL = '''
# ---------------------------------------------------------------------------
# 1. Imports & reproducibility
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from common.data import (SEED, IMG_SIZE, BATCH_SIZE, LABEL_MAP,
                         get_data, build_datasets, collect_labels,
                         split_fingerprint, make_synthetic_cell_dataset)
from common.eval import (evaluate_model, evaluate_metrics, Metrics,
                         collect_predictions,
                         final_evaluation_figure, plot_training_curves,
                         plot_confusion_matrix, plot_roc_curve,
                         find_misclassifications, plot_misclassifications)
from common.tb import ExperimentLogger, make_run_name, zip_logs_for_drive

tf.keras.utils.set_random_seed(SEED)   # seeds python/numpy/TF together

MODEL_NAME  = "{model_name}"     # prefix for every TensorBoard run name
MEMBER_NAME = "{member_name}"
print(f"Member notebook ready: {{MODEL_NAME}} ({{MEMBER_NAME}}), seed={{SEED}}")
'''

DATA_CELL = '''
# ---------------------------------------------------------------------------
# 2. Data: download once, cache in Drive, build the GROUP-IDENTICAL split.
# All three members run this cell and must print the same fingerprint.
# ---------------------------------------------------------------------------
cell_images = get_data(drive_data_dir=DRIVE_DATA)      # 27,558 NIH images

train_ds, val_ds, test_ds = build_datasets(
    cell_images, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    validation_fraction=0.30, seed=SEED, augment=False)

print("train fingerprint:", split_fingerprint(train_ds))
print("val   fingerprint:", split_fingerprint(val_ds))
print("test  fingerprint:", split_fingerprint(test_ds))
'''

PEEK_CELL = '''
# Peek at one batch — label convention check BEFORE any training:
# 1 = Parasitized (infected, positive class), 0 = Uninfected.
for x, y in train_ds.take(1):
    print("batch:", x.shape, "labels:", y.numpy().ravel()[:8])
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.2))
    for ax, img, lab in zip(axes, x[:6], y[:6]):
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title("Parasitized" if int(lab) == 1 else "Uninfected", fontsize=9)
        ax.axis("off")
    plt.show()
'''

EDA_SYNTH_CELL = '''
# Local dry run (no Colab needed): a tiny synthetic stand-in with the same
# folder layout and label convention, so the FULL flow below executes
# end-to-end even before the real dataset is downloaded.
_synth = make_synthetic_cell_dataset("data/_demo", per_class=12,
                                     img_size=(64, 64), seed=SEED)
demo_train, demo_val, demo_test = build_datasets(
    _synth, img_size=(64, 64), batch_size=8, seed=SEED)
print("demo split ready — same pipeline as the real dataset")
'''

# ===========================================================================
# Member 1 — Custom ResNet (assignment §3.1)
# ===========================================================================

RESNET_BUILDER_CELL = '''
# ---------------------------------------------------------------------------
# 3. The custom ResNet — Keras Subclassing API (assignment §3.1).
# Implemented in common/models.py and verified by tests; shown here for
# the architecture story (notebook ownership = understand every line).
# ---------------------------------------------------------------------------
from common.models import CustomResNet, ResidualBlock, build_custom_resnet

def make_resnet(spec: dict):
    """Build one experiment variant. `spec` maps straight onto the
    architecture/training levers we chose to study."""
    return build_custom_resnet(
        img_size=IMG_SIZE,
        dropout=spec.get("dropout", 0.3),
        l2=spec.get("l2", 0.0),
        width_mult=spec.get("width_mult", 1.0),
        blocks_per_stage=spec.get("blocks_per_stage", 2),
        bn_momentum=spec.get("bn_momentum", 0.99),
    )

model = make_resnet({})
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
print(f"CustomResNet parameters: {{model.count_params():,}}")
'''

RESNET_PROOF_CELL = '''
# Proof that this is a genuine subclassing-API model (rubric §3.1):
print("is keras.Model subclass :", isinstance(model, tf.keras.Model))
print("is Sequential           :", isinstance(model, tf.keras.Sequential))
print("has explicit call()     :", callable(getattr(CustomResNet, "call")))
print("residual blocks         :", len(model.stage1 + model.stage2 + model.stage3))
print("projection shortcuts    :",
      sum(1 for s in (model.stage2, model.stage3) for b in s if b.proj is not None))
print("grad-cam tap populated  :", model._last_conv_features is not None
      or "populated on first forward pass")
'''

RESNET_EXPERIMENTS_CELL = '''
# ---------------------------------------------------------------------------
# 4. The experiment campaign — 7 documented runs (rubric §4.1: meaningful
# changes only, clear progression from baseline to final).
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    dict(id=1, name="baseline",
         config=dict(),
         lr=1e-3,
         note="Reference run: default width/depth, no augmentation."),
    dict(id=2, name="bn_momentum_0.9",
         config=dict(bn_momentum=0.9),
         lr=1e-3,
         note="Lower BN momentum -> moving statistics track the batch more "
              "closely; tests stability under augmentation-free training."),
    dict(id=3, name="width_mult_0.5",
         config=dict(width_mult=0.5),
         lr=1e-3,
         note="Half-width network: how much capacity does the task need?"),
    dict(id=4, name="deeper_augmented",
         config=dict(blocks_per_stage=3, augment=True),
         lr=1e-3,
         note="Extra residual block per stage + flip/rot/zoom augmentation."),
    dict(id=5, name="dropout_0.5",
         config=dict(dropout=0.5),
         lr=1e-3,
         note="Stronger head regularisation for the overfitting gap."),
    dict(id=6, name="l2_1e-4",
         config=dict(l2=1e-4),
         lr=1e-3,
         note="Weight decay on every convolution kernel (He et al. recipe)."),
    dict(id=7, name="final_lr_2e-4",
         config=dict(dropout=0.3, l2=1e-4, augment=True),
         lr=2e-4,
         note="Best-of combination at a lower learning rate."),
]
assert len(EXPERIMENTS) >= 7
for e in EXPERIMENTS:
    print(make_run_name(MODEL_NAME, e["id"], e["name"]))
'''

RUNNER_CELL = '''
# ---------------------------------------------------------------------------
# 5. The experiment runner — one function, used for all 7 runs.
# Every run writes a TensorBoard directory whose NAME carries the experiment
# id and description, and whose HPARAMS record carries the configuration —
# so every claim in the report is traceable to a logged run (rubric §4.2).
# ---------------------------------------------------------------------------
from common.models import build_custom_resnet  # direct import for the runner

def run_experiment(spec: dict, datasets=None, epochs: int = 30,
                   logdir: str | None = None, verbose: int = 1):
    """Train one experiment variant and log everything the rubric requires."""
    train_ds, val_ds, test_ds = datasets

    exp = ExperimentLogger(
        model_name=MODEL_NAME, exp_id=spec["id"], description=spec["name"],
        hparams=dict(lr=spec["lr"], optimizer="adam",
                     batch_size=BATCH_SIZE, epochs=epochs,
                     augmentation=spec.get("augment", False),
                     dropout=spec.get("dropout", 0.3),
                     l2=spec.get("l2", 0.0),
                     width_mult=spec.get("width_mult", 1.0),
                     blocks_per_stage=spec.get("blocks_per_stage", 2),
                     bn_momentum=spec.get("bn_momentum", 0.99),
                     strategy="scratch",                       # from-scratch
                     model="CustomResNet"),
        logdir=logdir)

    augment = spec.get("augment", False)
    if augment:
        # rebuild only the train split with augmentation enabled
        tr, va, te = build_datasets(cell_images, img_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE, seed=SEED,
                                    augment=True)
        train_ds = tr
    model = make_resnet(spec)
    model.compile(optimizer=tf.keras.optimizers.Adam(spec["lr"]),
                  loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=exp.callbacks(), verbose=verbose)

    metrics, y_true, y_prob = evaluate_model(model, test_ds)
    exp.log_final_metrics(metrics.as_row())
    exp.log_hparams_once(metrics.as_row())
    print(f"[{{exp.run_name}}] {{metrics.format()}}")
    return dict(run=exp.run_name, model=model, history=history,
                metrics=metrics, y_true=y_true, y_prob=y_prob,
                spec=spec, note=spec["note"])


def result_to_row(r: dict) -> dict:
    """One row for the report's per-experiment metrics table (§5.1)."""
    row = dict(experiment=r["run"])
    row.update(r["metrics"].as_row())
    return row
'''

SMOKE_DEMO_CELL = '''
# ---------------------------------------------------------------------------
# 6. Smoke run — the FULL loop on the tiny demo dataset (seconds, not hours).
# Confirms: builder -> fit -> evaluate -> TensorBoard artifacts, before we
# spend GPU hours on the real campaign. On Colab with the real data, replace
# `datasets=demo` with `datasets=(train_ds, val_ds, test_ds)` and epochs=30.
# ---------------------------------------------------------------------------
demo = (demo_train, demo_val, demo_test)
demo_spec = dict(id=0, name="demo_smoke", config=dict(width_mult=0.25), lr=1e-3)
demo_run = run_experiment(demo_spec, datasets=demo, epochs=3, verbose=1)

_demo_dir = Path(os.environ["TB_LOGDIR"]) / demo_run["run"]
print("TensorBoard artifacts written:")
for p in sorted(_demo_dir.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(_demo_dir))
print("\\nMetrics row for the report table:")
import json
print(json.dumps(result_to_row(demo_run), indent=2, default=str))
'''

RESNET_FINAL_CELL = '''
# ---------------------------------------------------------------------------
# 7. FINAL MODEL — retrain the winning configuration with the full budget
# (rubric §5.2 visualizations are for the final version of each model).
# On Colab: pick the best experiment id from section 6 and run e.g.
#     final = run_experiment(EXPERIMENTS[6], datasets=(train_ds, val_ds, test_ds),
#                            epochs=40, logdir=os.environ["TB_LOGDIR"])
# Here we execute the glue with the demo-scale data so the notebook is
# runnable end-to-end; history is loaded from a checkpoint in a real run.
# ---------------------------------------------------------------------------
final_spec = EXPERIMENTS[6]
final = run_experiment(dict(final_spec, name="final"), datasets=demo,
                       epochs=3, verbose=1)

def print_history(history):
    """Epoch-by-epoch summary — the numbers quoted in the report prose."""
    for i, (l, a, vl, va) in enumerate(zip(history.history["loss"],
                                           history.history["accuracy"],
                                           history.history["val_loss"],
                                           history.history["val_accuracy"]), 1):
        print(f"epoch {{i:02d}}  loss={{l:.4f}}  acc={{a:.4f}}  "
              f"val_loss={{vl:.4f}}  val_acc={{va:.4f}}")

print_history(final["history"])
'''

FINAL_EVAL_CELL = '''
# ---------------------------------------------------------------------------
# 8. Final evaluation: the rubric's four plots in ONE row + full metrics.
# ---------------------------------------------------------------------------
fig, m, y_true, y_prob = final_evaluation_figure(
    final["model"], final["history"], demo_test,
    save_path="figures/" + MODEL_NAME + "_final_evaluation.png")
plt.show()
print(m.format())
print("\\nWHO naming:  sensitivity = recall(infected) ="
      f" {{m.sensitivity:.4f}}   specificity = {{m.specificity:.4f}}")
print("Targets — custom from-scratch: sensitivity >= 0.90, specificity >= 0.90")
'''

GRADCAM_CELL = '''
# ---------------------------------------------------------------------------
# 9. Explainability (rubric §5.4): Grad-CAM on the subclassed model.
# common/gradcam.py implements the eager GradientTape path that reads the
# feature map our model stashes on `model._last_conv_features` — the exact
# §5.4 requirement that encapsulated models fail. Heatmaps are computed on
# the LOGIT (pre-sigmoid) because a saturated sigmoid kills gradients.
# ---------------------------------------------------------------------------
from common.gradcam import (make_gradcam_heatmap, overlay_heatmap,
                            pick_gradcam_examples, gradcam_report_grid)

y_true_all, y_prob_all = collect_predictions(final["model"], demo_test)
images_all = np.concatenate([x.numpy() for x, _ in demo_test])

examples = pick_gradcam_examples(y_true_all, y_prob_all, images_all)
print("picked:", ["correct-infected", "correct-uninfected", "incorrect"][:len(examples)])

fig = gradcam_report_grid(examples, final["model"],
                          save_path="figures/" + MODEL_NAME + "_gradcam.png")
plt.show()
'''

ERROR_CELL = '''
# ---------------------------------------------------------------------------
# 10. Error analysis (rubric §5.3): misclassifications + fit diagnosis.
# ---------------------------------------------------------------------------
imgs, trues, preds, probs = find_misclassifications(final["model"], demo_test, n=3)
fig = plot_misclassifications(imgs, trues, preds, probs)
plt.show()

def diagnose_fit(history):
    """Simple overfitting/underfitting read-out from the curves."""
    h = history.history
    best = int(np.argmin(h["val_loss"]))
    gap = h["accuracy"][-1] - h["val_accuracy"][-1]
    verdict = []
    if gap > 0.05:
        verdict.append(f"OVERFITTING sign: train-val accuracy gap {{gap:.3f}}")
    if min(h["val_loss"]) > 0.693:
        verdict.append("UNDERFITTING sign: val_loss never beat the 0.693 chance line")
    if best < len(h["val_loss"]) - 2:
        verdict.append(f"early stop candidate: best val epoch {{best + 1}}")
    return "; ".join(verdict) or "no strong over/underfitting signal"

print("diagnosis:", diagnose_fit(final["history"]))
'''

CSV_SCAN_CELL = '''
# ---------------------------------------------------------------------------
# 11. Per-experiment table straight from the TensorBoard run folders —
# paste-ready for the report and a backup if TensorBoard is unavailable.
# ---------------------------------------------------------------------------
import csv

def scan_runs(logdir=None):
    logdir = Path(logdir or os.environ["TB_LOGDIR"])
    rows = []
    for d in sorted(logdir.iterdir()):
        f = d / "final_metrics.txt"
        if f.exists():
            row = dict(run=d.name)
            row.update(line.split(": ", 1) for line in f.read_text().splitlines())
            rows.append(row)
    return rows

for row in scan_runs():
    print(row)
# with open("results_table.csv", "w", newline="") as fh:
#     w = csv.DictWriter(fh, fieldnames=list(scan_runs()[0].keys())); w.writerows(...)
'''

# ===========================================================================
# Shared sections for the two transfer-learning members (assignment §3.2/3.3)
# ===========================================================================

TF_BUILDER_CELL_TMPL = '''
# ---------------------------------------------------------------------------
# 3. Transfer-learning model — {arch_title} (assignment {arch_ref}).
# Pretrained ImageNet backbone + GAP + Dense head. The backbone's own
# preprocessing layer stays INSIDE the model, so the data pipeline feeds
# raw 0-255 RGB and double-preprocessing bugs are impossible.
# ---------------------------------------------------------------------------
from common.models import build_transfer_model, unfreeze_top_n

def make_transfer_model(spec: dict):
    """One experiment variant. Frozen backbone = exp_01 baseline;
    `unfreeze_last: n` fine-tunes only the top n backbone layers."""
    model = build_transfer_model(
        MODEL_NAME, img_size=IMG_SIZE,
        dropout=spec.get("dropout", 0.3),
        dense_units=spec.get("dense_units", 128),
        freeze_backbone="unfreeze_last" not in spec,
        l2=spec.get("l2", 0.0),
    )
    if "unfreeze_last" in spec:
        unfreeze_top_n(model, n=spec["unfreeze_last"])   # call BEFORE compile
    return model

model = make_transfer_model(dict())
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
n_total = model.count_params()
n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print(f"{{model.name}}: {{n_total:,}} params total, {{n_train:,}} trainable")
'''

TF_PROOF_CELL = '''
# Sanity checks specific to transfer learning:
base = getattr(model, "_malaria_backbone")
print("backbone            :", base.name)
print("backbone trainable  :", base.trainable)
print("head layers         :", [l.name for l in model.layers[-3:]])
# The preprocessing is part of the graph — confirm:
print("preprocessing inside:", any("preprocess" in l.name for l in model.layers))
'''

def tf_experiments_cell(backbone: str, experiments: str) -> str:
    return (
        "# ---------------------------------------------------------------------------\n"
        f"# 4. The experiment campaign — 7 documented runs for {backbone}\n"
        "# (rubric §4.1: frozen -> head tuning -> progressive unfreezing).\n"
        "# ---------------------------------------------------------------------------\n"
        "EXPERIMENTS = [\n"
        + experiments +
        "]\n"
        "assert len(EXPERIMENTS) >= 7\n"
        "for e in EXPERIMENTS:\n"
        "    print(make_run_name(MODEL_NAME, e[\"id\"], e[\"name\"]))\n"
    )

TF_RUNNER_CELL = '''
# ---------------------------------------------------------------------------
# 5. The experiment runner — logs every rubric-required field per run.
# ---------------------------------------------------------------------------
def run_experiment(spec: dict, datasets=None, epochs: int = 25,
                   logdir: str | None = None, verbose: int = 1):
    """Train one experiment variant and log everything the rubric requires."""
    train_ds, val_ds, test_ds = datasets

    strategy = (f"unfreeze_last{{spec['unfreeze_last']}}"
                if "unfreeze_last" in spec else "frozen_head")
    exp = ExperimentLogger(
        model_name=MODEL_NAME, exp_id=spec["id"], description=spec["name"],
        hparams=dict(lr=spec["lr"], optimizer="adam",
                     batch_size=BATCH_SIZE, epochs=epochs,
                     augmentation=spec.get("augment", False),
                     dropout=spec.get("dropout", 0.3),
                     l2=spec.get("l2", 0.0),
                     dense_units=spec.get("dense_units", 128),
                     unfreeze_last=spec.get("unfreeze_last", 0),
                     strategy=strategy,
                     model=model.name),
        logdir=logdir)

    if spec.get("augment", False):
        tr, va, te = build_datasets(cell_images, img_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE, seed=SEED,
                                    augment=True)
        train_ds = tr
    m = make_transfer_model(spec)
    m.compile(optimizer=tf.keras.optimizers.Adam(spec["lr"]),
              loss="binary_crossentropy", metrics=["accuracy"])

    history = m.fit(train_ds, validation_data=val_ds, epochs=epochs,
                    callbacks=exp.callbacks(), verbose=verbose)

    metrics, y_true, y_prob = evaluate_model(m, test_ds)
    exp.log_final_metrics(metrics.as_row())
    exp.log_hparams_once(metrics.as_row())
    print(f"[{{exp.run_name}}] {{metrics.format()}}")
    return dict(run=exp.run_name, model=m, history=history,
                metrics=metrics, y_true=y_true, y_prob=y_prob,
                spec=spec, note=spec["note"])


def result_to_row(r: dict) -> dict:
    """One row for the report's per-experiment metrics table (§5.1)."""
    row = dict(experiment=r["run"])
    row.update(r["metrics"].as_row())
    return row
'''

TF_SMOKE_CELL = '''
# ---------------------------------------------------------------------------
# 6. Smoke run — full loop on the tiny demo dataset (frozen backbone, 3
# epochs). On Colab with real data: datasets=(train_ds, val_ds, test_ds).
# ---------------------------------------------------------------------------
demo = (demo_train, demo_val, demo_test)
demo_spec = dict(id=0, name="demo_smoke", lr=1e-3)
demo_run = run_experiment(demo_spec, datasets=demo, epochs=3, verbose=1)

_demo_dir = Path(os.environ["TB_LOGDIR"]) / demo_run["run"]
print("TensorBoard artifacts written:")
for p in sorted(_demo_dir.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(_demo_dir))
print("\\nMetrics row for the report table:")
import json
print(json.dumps(result_to_row(demo_run), indent=2, default=str))
'''

TF_FINAL_CELL = '''
# ---------------------------------------------------------------------------
# 7. FINAL MODEL — retrain the winning configuration with the full budget.
# On Colab: e.g.  final = run_experiment(EXPERIMENTS[6],
#                    datasets=(train_ds, val_ds, test_ds), epochs=30)
# Here the glue runs at demo scale so the notebook executes end-to-end.
# ---------------------------------------------------------------------------
final_spec = EXPERIMENTS[6]
final = run_experiment(dict(final_spec, name="final"), datasets=demo,
                       epochs=3, verbose=1)

def print_history(history):
    """Epoch-by-epoch summary — the numbers quoted in the report prose."""
    for i, (l, a, vl, va) in enumerate(zip(history.history["loss"],
                                           history.history["accuracy"],
                                           history.history["val_loss"],
                                           history.history["val_accuracy"]), 1):
        print(f"epoch {{i:02d}}  loss={{l:.4f}}  acc={{a:.4f}}  "
              f"val_loss={{vl:.4f}}  val_acc={{va:.4f}}")

print_history(final["history"])
'''

# ===========================================================================
# Prose (markdown) — written per member, deliberately NOT interchangeable
# ===========================================================================

def md(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"

RESNET_TITLE_MD = md(r"""
    # Notebook 1 — Custom ResNet from Scratch (Keras Subclassing API)

    **Model 1 of 3 — owned by: <MEMBER NAME>** · Group 6 · Formative 2

    This notebook owns the group's **hand-implemented residual network**
    (assignment §3.1): a from-scratch CNN built with the Keras **Subclassing
    API** — custom `ResidualBlock(layers.Layer)` and `CustomResNet(keras.Model)`
    subclasses with an explicit `call()` — following He et al. (2016),
    *Deep Residual Learning for Image Recognition*. It is **not** a pretrained
    ResNet; every weight starts random.

    **Why a residual design for malaria cells:** infected cells differ from
    uninfected ones by small, local texture changes (parasite stains inside the
    red blood cell). Deep plain CNNs degrade on such tasks because gradients
    vanish through many layers; identity shortcuts let gradients bypass each
    block, so deeper stacks stay trainable (He et al. 2016, §3.2). Our design:
    stem → 3 residual stages (64/128/256 channels) → GAP head, with 1×1
    **projection shortcuts** wherever a shape change occurs.

    **Notebook map:** §2 data & group-identical split → §3 architecture &
    subclassing proof → §4 the 7-experiment campaign → §6 smoke run →
    §7 final model → §8 four required plots → §9 Grad-CAM → §10 error
    analysis → §11 results table → logbook.
""")

RESNET_DATA_MD = md(r"""
    ### 2.1 Why the fingerprint matters

    All three members run §2 and must get **byte-identical fingerprints**.
    If two notebooks trained on different splits, the group's model comparison
    in the report would be invalid — the rubric explicitly demands one
    coherent comparison. `build_datasets` splits **by file** (70/15/15, seed
    42) *before* shuffling batches, so every image appears in exactly one
    split. `get_data` caches the 700 MB NIH zip in Drive, so later sessions
    skip the download.
""")

RESNET_EDA_MD = md(r"""
    ### 2.2 Dataset understanding (EDA)

    On Colab, extend this section with: class counts (`Parasitized` vs
    `Uninfected`, expect 13,779 each), a grid of random cells from each class,
    image size/histogram stats, and a note on artifacts visible in the raw
    set (the NIH library images include some irregular/funky cells — worth
    flagging before Grad-CAM interpretation, since background/platelet
    artifacts can attract attention).

    Locally, the synthetic demo below keeps the whole notebook executable:
    dark blob = parasite signal, same 1/0 convention as the real labels.
""")

RESNET_ARCH_MD = md(r"""
    ### 3.1 Architecture decisions and their justification

    | Decision | Choice | Justification |
    |---|---|---|
    | Depth | 3 stages × 2 residual blocks | small 128×128 crops; deeper overfits 27k images quickly |
    | Width | 32→64→128→256 | He et al.'s pattern, scaled down; keeps params < 3 M for Colab GPU time |
    | Shortcut | identity when shapes match, 1×1 projection otherwise | He et al. 2016 §3.2 option B — required whenever stride or channels change |
    | Normalization | BN after every conv | stabilizes from-scratch training; momentum is an experiment lever (exp 02) |
    | Head | GAP → Dense(256) → Dropout → Dense(1, sigmoid) | GAP kills position overfitting; single sigmoid = binary diagnosis |
    | Loss | binary cross-entropy | balanced classes; probability output feeds ROC-AUC |

    **Grad-CAM readiness (assignment §5.4):** `call()` stores the last
    convolutional feature map on `self._last_conv_features`, and
    `common/gradcam.py` reads exactly that tensor under `GradientTape`.
    The §5.4 note warns that encapsulated models fail here — ours cannot,
    because the tap is part of the design (verified by
    `tests/test_models_gradcam.py::test_gradcam_heatmap_localizes_blob`).
""")

RESNET_PROOF_MD = md(r"""
    The printed checks are the evidence the rubric's §3.1 asks for: a genuine
    `keras.Model` **subclass** (not Sequential, not `applications.ResNet50`),
    with an explicit `call()`, real residual blocks, and projection shortcuts
    wherever shapes change. Parameter count goes in the report's methodology
    table alongside the two transfer-learning models.
""")

RESNET_EXP_MD = md(r"""
    ### 4.1 Experiment design — what each run tests and why

    The campaign is a *progression*, not scattered attempts (rubric §4.1):
    capacity first (exp 2–4), then regularization (exp 5–6), then the
    combined final recipe (exp 7). Every hypothesis below was written
    **before** the run and revisited in the logbook (§13).

    | run | variable changed | hypothesis |
    |---|---|---|
    | exp_01_baseline | — (reference) | establishes the training behaviour we compare against |
    | exp_02_bn_momentum_0.9 | BN momentum 0.99→0.9 | faster-moving statistics → smoother val curves |
    | exp_03_width_mult_0.5 | half width | the task may not need 2.7 M params; smaller = faster + less overfit |
    | exp_04_deeper_augmented | +1 block/stage + augmentation | more capacity only helps once augmentation fights the overfitting it invites |
    | exp_05_dropout_0.5 | head dropout 0.3→0.5 | shrinks the train/val gap seen in exp_01 |
    | exp_06_l2_1e-4 | L2 on conv kernels | weight decay as an alternative regularizer |
    | exp_07_final | best combination @ lr 2e-4 | lower LR for the final polish, combining what worked |

    Each run's full configuration is logged in its TensorBoard HPARAMS record
    and its `training_log.csv` — the report cites run names, not vibes.
""")

RUNNER_MD = md(r"""
    ### 5.1 What the runner guarantees

    One function = one experiment, so nothing drifts between runs. Each run
    writes: (1) a TensorBoard directory named
    `{model}_exp_{id}_{description}` — the rubric's naming convention,
    enforced by `make_run_name`; (2) an HPARAMS record (lr, optimizer, batch
    size, epochs, augmentation, dropout, L2, strategy) for the hparams
    dashboard comparison screenshots; (3) `training_log.csv` as extra
    evidence; (4) final test metrics *into the same run directory* via
    `log_final_metrics`, so run-to-run comparisons in TensorBoard include
    the end-of-training evaluation.

    On Colab, `TB_LOGDIR` points into Drive (set in §0), so a disconnected
    session cannot eat 30 epochs of evidence.
""")

SMOKE_MD = md(r"""
    ### 6.1 Reading the smoke run

    Three demo epochs on 24 synthetic images should already beat 0.5 val
    accuracy — the blob signal is easy. What we are really checking:

    1. **No crashes in the full glue** — builder → fit → evaluate → log.
    2. **TensorBoard artifacts exist** — the listing proves where evidence
       lives before the real campaign runs.
    3. **Metrics row is complete** — the JSON line is exactly one row of the
       report's §5.1 table (accuracy, precision, recall/sensitivity,
       specificity, F1, ROC-AUC).

    Expected on the real dataset: exp_01 baseline lands around the high-80s
    accuracy after ~30 epochs (published custom CNNs on this exact NIH set
    reach 89–97% sensitivity); our target bar for custom from-scratch models
    is ≥ 0.90 sensitivity and ≥ 0.90 specificity (assignment §5.1).
""")

FINAL_MD = md(r"""
    ### 7.1 From experiments to the final model

    The final version is exp_07's recipe retrained with the full epoch budget
    and the learning-rate drop that late-training curves justified. In a real
    session this cell loads the checkpoint of the best run and (optionally)
    retrains with more epochs. `print_history` gives the epoch-by-epoch
    numbers the report prose quotes — e.g. "validation loss plateaued at
    epoch N near X" must be traceable to this table *and* the TensorBoard
    scalars for the same run.
""")

FINAL_EVAL_MD = md(r"""
    ### 8.1 Interpreting the four plots (write this in your own words after
    the real run — prompts, not final prose)

    - **Accuracy curve** — does validation accuracy track training, or split
      apart (overfitting)? Where does it plateau?
    - **Loss curve** — the honest plot: loss exposes what accuracy hides.
      A rising val loss with flat val accuracy = memorization starting.
    - **Confusion matrix** — for malaria, the top-left vs top-right asymmetry
      *is* the story: false negatives (infected → uninfected) are the costly
      error. State FN and FP counts explicitly.
    - **ROC curve** — threshold-free view of the same trade-off. The AUC
      number in §11 and this curve are the same evidence at two zoom levels;
      explain what moving the operating threshold does to sensitivity vs
      specificity, and where WHO-style priorities would put it.
""")

GRADCAM_MD = md(r"""
    ### 9.1 Interpreting the heatmaps (prompts — answer after the real run)

    - Does attention sit **inside the cell** or on background? Infected-cell
      explanations should concentrate on intra-cell structure.
    - Compare the **custom ResNet** against the transfer-learning notebooks:
      a from-scratch model often spreads attention differently than
      ImageNet-pretrained features — that contrast is a report paragraph.
    - The **incorrect** example: did the model look at a plausible region and
      get fooled by an artifact, or look somewhere meaningless? Either answer
      is report gold — it questions the model's reasoning instead of parroting
      its accuracy.
    - Grad-CAM shows *where* the model looked, never *whether that looking is
      medically valid* — keep the interpretation critical (rubric language).
""")

ERROR_MD = md(r"""
    ### 10.1 Error analysis — how to write it up

    For each mined misclassification: **true vs predicted label + the model's
    confidence**, then a plausible cause (stain artifact, late-stage vs early
    parasite, ambiguous cell, background clutter). The `diagnose_fit` readout
    is deliberately simple — the report should cite the *specific* curve
    evidence (epoch numbers, gap sizes) and what was tried in response
    (which experiment id came out of which diagnosis).
""")

CSV_MD = md(r"""
    ### 11. Results table & TensorBoard evidence

    `scan_runs()` rebuilds the per-experiment metrics table from the run
    folders themselves — the same numbers TensorBoard shows, in CSV-ready
    form. Paste it into the report (one table per model, one row per
    experiment, run names exactly as logged). For the appendix: export the
    Drive `logs/fit` directory (zip helper below) and screenshot the hparams
    dashboard comparison — that is the rubric's "TensorBoard evidence showing
    run-to-run comparisons".
""")

DEFENSE_MD = md(r"""
    ### 12. Oral defense preparation (10–15 min: 2-min overview + Q&A)

    Expect to be asked: *why residual connections here?* (gradient flow,
    He et al.), *why this depth/width?* (exp 03/04 evidence), *what did BN
    momentum change?* (exp 02 curves), *where does the model look and do you
    trust it?* (§9), *what is your model's worst error and why?* (§10),
    *where does it rank among the group's three models and what evidence says
    so?* (§11 table + ROC trade-offs, not accuracy alone).
""")

LOGBOOK_MD = md(r"""
    ### 13. Lab journal (write as you go — this becomes the report's voice)

    After each experiment, four lines: **what I changed · what I expected ·
    what happened · what surprised me**. Example shape (do not copy):

    > exp_03 (width 0.5): expected a small accuracy drop; got −0.8 pts and a
    > *tighter* train/val gap. Surprised: half the parameters still beat the
    > baseline's validation loss by epoch 12.

    Honest negative results are rubric gold: the campaign must read as a
    progression of *decisions*, and decisions come from expectations that
    failed as well as succeeded.
""")

# --- transfer-learning prose -------------------------------------------------

def tf_title_md(arch: str, arch_ref: str, owner: str, why: str) -> str:
    return md(f"""
        # Notebook {owner[0]} — {arch} Transfer Learning

        **Model {owner[0]} of 3 — owned by: {owner[1]}** · Group 6 · Formative 2

        This notebook owns one of the group's two **transfer-learning models**
        (assignment {arch_ref}): a pretrained **{arch}** backbone (ImageNet)
        with a fresh classification head, fine-tuned on the NIH malaria cell
        set.

        {why}

        **Notebook map:** §2 data & group-identical split → §3 architecture
        & sanity checks → §4 the 7-experiment campaign → §6 smoke run →
        §7 final model → §8 four required plots → §9 Grad-CAM → §10 error
        analysis → §11 results table → logbook.
    """)

TF_DATA_MD = RESNET_DATA_MD  # same split story, shared by the whole group

TF_ARCH_MD_T = md(r"""
    ### 3.1 Architecture decisions and their justification

    | Decision | Choice | Justification |
    |---|---|---|
    | Backbone | {arch} (ImageNet) | {backbone_why} |
    | Head | GAP → Dropout → Dense(1, sigmoid) | GAP aggregates 4×4×C feature maps; single sigmoid feeds ROC-AUC |
    | Preprocessing | backbone's own `preprocess_input` **inside** the model graph | the pipeline stays raw-RGB end-to-end — double-preprocessing bugs are structurally impossible |
    | Phase 1 (exp 01–05) | backbone frozen, train head only | frozen features + small head train fast and give a clean baseline |
    | Phase 2 (exp 06+) | unfreeze last N layers at 10× lower LR | discriminative fine-tuning: high-level ImageNet features adapt to cells without destroying the backbone |

    **Grad-CAM readiness:** the backbone's final conv output is exposed by
    name, so `common/gradcam.py` can tap it directly (functional path).
""")

TF_PROOF_MD = md(r"""
    The printed checks are the transfer-learning specifics the defense will
    probe: how many parameters are actually trainable (frozen phase), that
    the backbone's preprocessing is part of the graph, and which head layers
    are new. Fine-tuning experiments re-run §3 via `make_transfer_model(spec)`
    with `unfreeze_last: N` — trainable-parameter counts then change, and the
    hparams record captures the strategy per run.
""")

TF_EXP_MD = md(r"""
    ### 4.1 Experiment design — what each run tests and why

    The campaign mirrors the transfer-learning decision path (rubric §4.1):
    establish the frozen baseline, tune the head, then control what
    fine-tuning costs before unlocking the backbone.

    {table}

    Each run's configuration is logged in its TensorBoard HPARAMS record and
    `training_log.csv`; the report cites run names exactly as printed above.
""")

TF_RUNNER_MD = RUNNER_MD  # same logging guarantees

TF_SMOKE_MD = md(r"""
    ### 6.1 Reading the smoke run

    Three demo epochs with a frozen ImageNet backbone on 24 synthetic images:
    the point is the glue, not the number. Checks: (1) the head trains while
    backbone weights stay frozen (hparams `strategy=frozen_head`), (2) TensorBoard
    artifacts land in the expected run directory, (3) the JSON metrics row is
    complete for the report table.

    Expected on the real dataset: frozen {arch} heads on this NIH set
    typically land high-80s accuracy; published transfer-learning baselines
    on the same images report ~89% sensitivity / ~95% specificity
    (Rajaraman et al. 2018). Our target bar: ≥ 0.95 sensitivity and
    ≥ 0.90 specificity (assignment §5.1).
""")

TF_FINAL_MD = FINAL_MD
TF_EVAL_MD = FINAL_EVAL_MD
TF_GRADCAM_MD = md(r"""
    ### 9.1 Interpreting the heatmaps (prompts — answer after the real run)

    - Does the pretrained backbone attend **inside the cell** or to
      background/stain? ImageNet features sometimes latch onto global color
      statistics — that would be a shortcut worth flagging.
    - Compare against the custom ResNet notebook: pretrained features often
      produce sharper, more object-like attention; a from-scratch model may
      spread attention. Explain what that means for trust.
    - The **incorrect** example: did attention follow the artifact that
      fooled the model? Grad-CAM on errors is how the rubric separates
      description from analysis.
""")
TF_ERROR_MD = ERROR_MD
TF_CSV_MD = CSV_MD
TF_DEFENSE_MD = md(r"""
    ### 12. Oral defense preparation (10–15 min: 2-min overview + Q&A)

    Expect: *why {arch} and not the other backbone?* (cite the group's
    comparison, not vibes), *what exactly did fine-tuning change?* (exp 06
    curves + trainable counts), *frozen vs unfrozen — what did each cost and
    buy?* (§11 table), *what does the ROC trade-off say about your operating
    threshold?* (§8), *where does the model look — cell or stain — and do you
    trust it?* (§9), *where does it rank among the three models?* (evidence,
    not accuracy alone).
""")

TF_LOGBOOK_MD = LOGBOOK_MD

# ===========================================================================
# Experiment tables for the two transfer-learning members
# ===========================================================================

MOBILENET_EXPERIMENTS = '''    dict(id=1, name="baseline_frozen", lr=1e-3,
         note="Frozen backbone, default head: the reference point."),
    dict(id=2, name="dropout_0.5", lr=1e-3, dropout=0.5,
         note="Stronger head regularisation vs the baseline gap."),
    dict(id=3, name="dense_256", lr=1e-3, dense_units=256,
         note="Bigger head: can a wider classifier read MobileNet features better?"),
    dict(id=4, name="augmented", lr=1e-3, augment=True,
         note="Flip/rot/zoom on the frozen-backbone setup."),
    dict(id=5, name="head_lr_1e-4", lr=1e-4,
         note="Lower LR for the head: gentler convergence, same capacity."),
    dict(id=6, name="unfreeze_last20", lr=1e-5, unfreeze_last=20, dropout=0.3,
         note="Fine-tune top-20 layers at 10x lower LR (discriminative FT)."),
    dict(id=7, name="final", lr=1e-5, unfreeze_last=20, augment=True, dropout=0.3,
         note="Best-of combination: unfreeze + augmentation."),
'''

DENSENET_EXPERIMENTS = '''    dict(id=1, name="baseline_frozen", lr=1e-3,
         note="Frozen DenseNet121, default head: the reference point."),
    dict(id=2, name="dropout_0.5", lr=1e-3, dropout=0.5,
         note="Stronger head regularisation vs the baseline gap."),
    dict(id=3, name="augmented", lr=1e-3, augment=True,
         note="Flip/rot/zoom on the frozen-backbone setup."),
    dict(id=4, name="head_lr_1e-4", lr=1e-4,
         note="Lower LR for the head: gentler convergence, same capacity."),
    dict(id=5, name="unfreeze_last40", lr=1e-5, unfreeze_last=40,
         note="Shallow fine-tune: only the last dense block adapts."),
    dict(id=6, name="unfreeze_last120", lr=1e-5, unfreeze_last=120, dropout=0.3,
         note="Deeper fine-tune: three dense blocks adapt — watch for overfitting."),
    dict(id=7, name="final", lr=1e-5, unfreeze_last=40, augment=True, dropout=0.3,
         note="Best-of combination at the stable fine-tuning depth."),
'''

MOBILENET_EXP_TABLE = """| run | variable changed | hypothesis |
|---|---|---|
| exp_01_baseline_frozen | — (reference) | how far do frozen ImageNet features already go? |
| exp_02_dropout_0.5 | head dropout 0.3→0.5 | shrink the head's train/val gap |
| exp_03_dense_256 | head 128→256 units | more head capacity to read the 1280-d feature vector |
| exp_04_augmented | + flip/rot/zoom | augmentation before touching the backbone |
| exp_05_head_lr_1e-4 | head LR 1e-3→1e-4 | gentler convergence; stability reference for fine-tuning |
| exp_06_unfreeze_last20 | top-20 layers trainable @ 1e-5 | discriminative fine-tuning buys task-specific features cheaply |
| exp_07_final | exp_06 + augmentation | the combined recipe we defend as final |"""

DENSENET_EXP_TABLE = """| run | variable changed | hypothesis |
|---|---|---|
| exp_01_baseline_frozen | — (reference) | how far do frozen ImageNet features already go? |
| exp_02_dropout_0.5 | head dropout 0.3→0.5 | shrink the head's train/val gap |
| exp_03_augmented | + flip/rot/zoom | augmentation before touching the backbone |
| exp_04_head_lr_1e-4 | head LR 1e-3→1e-4 | gentler convergence; stability reference for fine-tuning |
| exp_05_unfreeze_last40 | last dense block trainable @ 1e-5 | shallow fine-tuning: minimal risk, measurable gain |
| exp_06_unfreeze_last120 | three dense blocks trainable @ 1e-5 | deeper fine-tuning: more gain, more overfitting risk |
| exp_07_final | exp_05 + augmentation | the combined recipe we defend as final |"""

# ===========================================================================
# Notebook assembly
# ===========================================================================

def fmt(text: str, **kw) -> str:
    return textwrap.dedent(text).format(**kw)

def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def build_resnet_notebook() -> tuple[str, dict]:
    member = ("1", "<MEMBER 1 NAME>")
    cells = [
        md_cell(RESNET_TITLE_MD),
        code_cell(fmt(ONE_SCRIPT, model_name="custom_resnet")),
        code_cell(fmt(SETUP_CELL, model_name="custom_resnet", member_name=member[1])),
        md_cell(RESNET_DATA_MD),
        code_cell(DATA_CELL),
        md_cell(RESNET_EDA_MD),
        code_cell(PEEK_CELL),
        code_cell(EDA_SYNTH_CELL),
        md_cell(RESNET_ARCH_MD),
        code_cell(RESNET_BUILDER_CELL.format("{}")),
        md_cell(RESNET_PROOF_MD),
        code_cell(RESNET_PROOF_CELL),
        md_cell(RESNET_EXP_MD),
        code_cell(RESNET_EXPERIMENTS_CELL),
        md_cell(RUNNER_MD),
        code_cell(RUNNER_CELL),
        md_cell(SMOKE_MD),
        code_cell(SMOKE_DEMO_CELL),
        md_cell(FINAL_MD),
        code_cell(RESNET_FINAL_CELL),
        md_cell(TF_EVAL_MD),
        code_cell(FINAL_EVAL_CELL),
        md_cell(TF_GRADCAM_MD),
        code_cell(GRADCAM_CELL),
        md_cell(ERROR_MD),
        code_cell(ERROR_CELL),
        md_cell(CSV_MD),
        code_cell(CSV_SCAN_CELL),
        md_cell(DEFENSE_MD),
        md_cell(LOGBOOK_MD),
    ]
    return "01_custom_resnet_member.ipynb", make_notebook(cells)


MOBILENET_WHY = md(r"""
    **Why MobileNetV2:** inverted residuals + linear bottlenecks make it the
    cheapest backbone in the group (≈3.5 M params), so we can run more
    experiments per GPU-hour and probe fine-tuning depth without waiting.
    Its ImageNet features transfer well to circular, textured objects —
    a reasonable prior for blood cells.
""")

DENSENET_WHY = md(r"""
    **Why DenseNet121:** dense connectivity concatenates features across
    depth, giving rich multi-scale texture cues for parasite staining and a
    strong contrast against MobileNetV2's efficiency-first design within the
    group. Heavier (≈8 M params backbone) but distinctive — the group's
    comparison gains a real family difference.
""")


def build_transfer_notebook(nb_idx: str, member_name: str, backbone: str,
                            arch_title: str, arch_ref: str, why_md: str,
                            backbone_why: str, experiments_src: str,
                            exp_table: str) -> tuple[str, dict]:
    data_md = TF_DATA_MD
    arch_md = fmt(TF_ARCH_MD_T, arch=arch_title, backbone_why=backbone_why)
    exp_md = fmt(TF_EXP_MD, table=exp_table)
    smoke_md = fmt(TF_SMOKE_MD, arch=arch_title)
    defense_md = fmt(TF_DEFENSE_MD, arch=arch_title)
    cells = [
        md_cell(tf_title_md(arch_title, arch_ref, (nb_idx, member_name), why_md)),
        code_cell(fmt(ONE_SCRIPT, model_name=backbone)),
        code_cell(fmt(SETUP_CELL, model_name=backbone, member_name=member_name)),
        md_cell(data_md),
        code_cell(DATA_CELL),
        md_cell(RESNET_EDA_MD),
        code_cell(PEEK_CELL),
        code_cell(EDA_SYNTH_CELL),
        md_cell(arch_md),
        code_cell(fmt(TF_BUILDER_CELL_TMPL, arch_title=arch_title,
                      arch_ref=arch_ref)),
        md_cell(TF_PROOF_MD),
        code_cell(TF_PROOF_CELL),
        md_cell(exp_md),
        code_cell(tf_experiments_cell(backbone, experiments_src)),
        md_cell(TF_RUNNER_MD),
        code_cell(TF_RUNNER_CELL),
        md_cell(smoke_md),
        code_cell(TF_SMOKE_CELL),
        md_cell(TF_FINAL_MD),
        code_cell(TF_FINAL_CELL),
        md_cell(TF_EVAL_MD),
        code_cell(FINAL_EVAL_CELL),
        md_cell(TF_GRADCAM_MD),
        code_cell(GRADCAM_CELL),
        md_cell(TF_ERROR_MD),
        code_cell(ERROR_CELL),
        md_cell(TF_CSV_MD),
        code_cell(CSV_SCAN_CELL),
        md_cell(defense_md),
        md_cell(TF_LOGBOOK_MD),
    ]
    return f"{nb_idx}_{backbone}_member.ipynb", make_notebook(cells)


# ===========================================================================
# Validation + main
# ===========================================================================

def validate_notebook(nb: dict, name: str) -> list[str]:
    """Schema-validate with nbformat and syntax-compile every code cell."""
    errors: list[str] = []
    try:
        import nbformat
        nbformat.validate(nb)
    except ImportError:
        errors.append(f"[{name}] nbformat not installed — schema check skipped")
    except Exception as e:
        errors.append(f"[{name}] nbformat validation failed: {e}")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            try:
                ast.parse(src)
            except SyntaxError as e:
                errors.append(f"[{name}] code cell {i}: SyntaxError: {e}")
    return errors


def main() -> int:
    notebooks = [
        build_resnet_notebook(),
        build_transfer_notebook(
            "02", "<MEMBER 2 NAME>", "mobilenetv2",
            "MobileNetV2", "(assignment §3.2)", MOBILENET_WHY,
            "inverted residuals + linear bottlenecks: the cheapest backbone in "
            "the group (~3.5 M params), enabling more experiments per GPU-hour",
            MOBILENET_EXPERIMENTS, MOBILENET_EXP_TABLE),
        build_transfer_notebook(
            "03", "<MEMBER 3 NAME>", "densenet121",
            "DenseNet121", "(assignment §3.3)", DENSENET_WHY,
            "dense cross-depth feature concatenation gives multi-scale texture "
            "cues for parasite staining — a deliberate family contrast to "
            "MobileNetV2",
            DENSENET_EXPERIMENTS, DENSENET_EXP_TABLE),
    ]

    failures: list[str] = []
    OUT_DIR.mkdir(exist_ok=True)
    for name, nb in notebooks:
        errs = validate_notebook(nb, name)
        if errs:
            failures.extend(errs)
            continue
        path = OUT_DIR / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"OK  {path.relative_to(PROJECT_ROOT)}  "
              f"({n_md} markdown / {n_code} code cells)")

    if failures:
        print("\nVALIDATION FAILURES:")
        for f in failures:
            print(" ", f)
        return 1
    print("\nAll notebooks generated and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

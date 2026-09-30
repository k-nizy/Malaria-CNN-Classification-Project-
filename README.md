# Malaria Diagnosis CNN + Transfer Learning

**Formative 2 Group Project — University Coursework**

---

## 1. Project Overview

This project builds and evaluates four Convolutional Neural Network models for automated malaria diagnosis from microscopic blood cell images. The dataset consists of 27,558 NIH/NLM cell images (Parasitized vs Uninfected), balanced across classes.

| Model | Type | Status |
|-------|------|--------|
| Custom ResNet | From-scratch (Keras Subclassing API) | ✅ Baseline defined |
| Transfer Model 1 | Transfer learning — Architecture TBD | 🔄 Placeholder |
| Transfer Model 2 | Transfer learning — Architecture TBD | 🔄 Placeholder |
| Additional Custom CNN | From-scratch (custom architecture) | 🔄 Placeholder |

**Team size:** 4 members  
**Experiments per model:** 7+ meaningful experiments  
**Total experiments:** 28+  
**Key deliverable:** Unified group report + oral defense

---

## 2. Project Objectives

- Implement 4 distinct CNN architectures for binary malaria classification
- Run at least 7 well-documented experiments per model (28+ total)
- Achieve ≥90% sensitivity and ≥90% specificity on held-out test set
- Produce reproducible results with identical data splits across all members
- Generate TensorBoard logs for every experiment
- Create all required visualizations: accuracy/loss curves, confusion matrix, ROC-AUC, Grad-CAM
- Write a single unified group report with critical discussion
- Prepare for oral defense

---

## 3. Team Structure

| Member | Assigned Model | Responsibilities |
|--------|----------------|------------------|
| Member 1 | **TBD** | Model implementation, 7+ experiments, TensorBoard, evaluation, Grad-CAM, report section, oral defense |
| Member 2 | **Custom ResNet** | Model implementation, 7+ experiments, TensorBoard, evaluation, Grad-CAM, report section, oral defense |
| Member 3 | **TBD** | Model implementation, 7+ experiments, TensorBoard, evaluation, Grad-CAM, report section, oral defense |
| Member 4 | **Additional Custom CNN** | Model implementation, 7+ experiments, TensorBoard, evaluation, Grad-CAM, report section, oral defense |

**Group Leader (Member 2):** Also coordinates shared repository, deadlines, evidence collection, report integration, quality control, final submission.

*Model assignments are flexible and can be reassigned — see [docs/team_responsibilities.md](docs/team_responsibilities.md)*

---

## 4. Current Model Structure

```
Model 1: Custom ResNet
├── From-scratch implementation using Keras Subclassing API
├── Residual blocks with identity/projection shortcuts (He et al. 2016)
├── Configurable: width_mult, blocks_per_stage, dropout, L2, BN momentum
└── Grad-CAM ready via self._last_conv_features tap

Model 2: Transfer Model 1 (TBD)
├── Pretrained backbone (ImageNet)
├── Frozen baseline → progressive unfreezing experiments
├── Architecture options: MobileNetV2, DenseNet121, EfficientNetB0, etc.
└── Decision deferred — see [docs/model_selection.md](docs/model_selection.md)

Model 3: Transfer Model 2 (TBD)
├── Different architecture from Transfer Model 1
├── Different fine-tuning strategy
└── Architecture options: (remaining from above, or ResNet50, VGG16, etc.)

Model 4: Additional Custom CNN
├── From-scratch, different architecture from Custom ResNet
├── Could be: smaller ResNet variant, DenseNet-style, EfficientNet-style, or novel
└── Must use Subclassing API for fair comparison with Model 1
```

---

## 5. Model Placeholders

**Transfer Model 1:** TBD  
**Transfer Model 2:** TBD  

We have NOT finalized these choices. They will be selected based on:
- Different architectural families (for meaningful comparison)
- Computational cost on Colab GPU
- Grad-CAM compatibility
- Fine-tuning flexibility
- Meaningful experiment space (7+ distinct changes)

See [docs/model_selection.md](docs/model_selection.md) for decision criteria.

---

## 6. Dataset

**Source:** NIH/NLM Malaria Dataset  
**URL:** `https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip`  
**Total images:** 27,558 (13,779 per class)  
**Classes:** `Parasitized` (label 1, positive/infected), `Uninfected` (label 0)  
**Split:** 70% train / 15% validation / 15% test (stratified by file, seed=42)  
**Image size:** 128×128 (default; transfer models may use 224×224)  
**Augmentation:** RandomFlip + RandomRotation(0.15) + RandomZoom(0.15)  

**Label convention (CRITICAL):**
- `1 = Parasitized = Infected = Positive class`
- `0 = Uninfected = Negative class`
- **Recall = Sensitivity** (on infected class)
- **Specificity = TNR** (on uninfected class)

All members use IDENTICAL splits via shared `common/data.py` pipeline. Verified by `split_fingerprint()`.

---

## 7. Project Requirements (from rubric)

| Criterion | Points | Key Requirements |
|-----------|--------|------------------|
| Problem Framing & Literature Review | 5 | >5 scholarly refs, critical (not descriptive) review |
| Model Implementation & Experimentation | 5 | 7+ experiments/model, TensorBoard evidence, justified choices |
| Evaluation, Results & Visual Analysis | 5 | Results tables, 4 plots/model, Grad-CAM + incorrect, critical interpretation |
| Code Quality & Documentation | 5 | Modular, reproducible, Markdown explaining what/why/how |
| Report Quality & Collaboration | 10 | Scholarly prose, correct citations, <20% AI text, integrated narrative |
| Presentation / Demo Video | 5 | Justify choices, interpret visuals live, answer follow-ups |

---

## 8. Experiment Requirements

**Every model gets 7+ experiments.** The baseline is **Experiment 01** for each model — this is NOT a fifth model.

| Exp ID | Purpose | Example Changes |
|--------|---------|-----------------|
| EXP-01 | Baseline (frozen backbone for transfer; default config for custom) | — |
| EXP-02 | Learning rate / optimizer | LR 1e-3 vs 1e-4, Adam vs SGD |
| EXP-03 | Regularization | Dropout, L2, label smoothing |
| EXP-04 | Architecture depth/width | width_mult, blocks_per_stage, head size |
| EXP-05 | Data augmentation | Stronger/weaker aug, mixup, cutmix |
| EXP-06 | Fine-tuning (transfer) / BN momentum (custom) | Unfreeze top-N, BN momentum |
| EXP-07 | Final best config | Combined best hyperparameters |
| EXP-08+ | Optional: ensembling, TTA, loss functions | — |

**Experiment design must depend on architecture.** Do not force identical changes across all models.

Each experiment documented in [docs/experiment_template.md](docs/experiment_template.md).

---

## 9. Evaluation Metrics (all required)

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy | — | Overall correctness |
| Precision | — | TP / (TP + FP) |
| **Recall / Sensitivity** | **≥90%** | TP / (TP + FN) — infected class |
| **Specificity** | **≥90%** | TN / (TN + FP) — uninfected class |
| F1-score | — | Harmonic mean of precision/recall |
| ROC-AUC | — | Area under ROC curve |

All metrics computed via `common.eval.evaluate_metrics()` (shared utility).

---

## 10. Explainability

- **Grad-CAM** for every final model (required by rubric)
- 3 examples per model: correct-infected, correct-uninfected, first incorrect
- Implementation in `common/gradcam.py` (works for subclassed AND functional models)
- Report grid: original image + heatmap overlay

---

## 11. Repository Structure

```
malaria-cnn-formative-2/
├── README.md
├── requirements.txt
├── .gitignore
├── IMPLEMENTATION_PLAN.md          # Detailed project plan (existing)
├── Malaria-Diagnosis_CNN_Group6-EvenNumber.ipynb  # Starter notebook (existing)
│
├── common/                         # Shared utilities (existing, DO NOT MODIFY)
│   ├── __init__.py
│   ├── data.py       # Data pipeline (download, split, augment, synthetic)
│   ├── eval.py       # Metrics + 4 required plots + error analysis
│   ├── gradcam.py    # Grad-CAM (subclassed + functional models)
│   ├── models.py     # CustomResNet + transfer model builders
│   └── tb.py         # TensorBoard experiment logging
│
├── src/                            # Public API layer (thin wrappers)
│   ├── data/           # Re-exports common.data
│   ├── preprocessing/  # Re-exports common.data augmentation
│   ├── evaluation/     # Re-exports common.eval metrics
│   ├── visualization/  # Re-exports common.eval plots
│   ├── explainability/ # Re-exports common.gradcam
│   └── utils/
│       └── project_config.py  # Model config + team assignment
│
├── notebooks/
│   ├── member_1/
│   │   └── model_training.ipynb  # Template for Member 1
│   ├── member_2/
│   │   └── model_training.ipynb  # Template for Member 2 (Custom ResNet)
│   ├── member_3/
│   │   └── model_training.ipynb  # Template for Member 3
│   ├── member_4/
│   │   └── model_training.ipynb  # Template for Member 4 (Custom CNN)
│   ├── 01_custom_resnet_member.ipynb    # Existing generated notebook
│   ├── 02_mobilenetv2_member.ipynb      # Existing generated notebook
│   └── 03_densenet121_member.ipynb      # Existing generated notebook
│
├── experiments/
│   ├── custom_resnet/      # Experiment logs/configs for Custom ResNet
│   ├── transfer_model_1/   # Experiment logs/configs for Transfer Model 1
│   ├── transfer_model_2/   # Experiment logs/configs for Transfer Model 2
│   └── custom_cnn/         # Experiment logs/configs for Custom CNN
│
├── logs/
│   └── tensorboard/
│       ├── custom_resnet/
│       ├── transfer_model_1/
│       ├── transfer_model_2/
│       └── custom_cnn/
│
├── results/
│   ├── metrics/              # CSV/JSON metric tables
│   ├── figures/              # Generated plots (accuracy, loss, CM, ROC)
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── error_analysis/       # Misclassification examples
│   └── gradcam/              # Grad-CAM heatmaps + report grids
│
├── models/                   # Saved model weights (.weights.h5)
│
├── report/
│   ├── report.md             # Unified group report (skeleton created)
│   ├── figures/              # Figures for report
│   └── references/           # BibTeX / reference manager files
│
├── docs/
│   ├── team_responsibilities.md
│   ├── experiment_template.md
│   ├── workflow.md
│   └── model_selection.md
│
├── scripts/                  # Build/debug utilities (existing)
│   ├── build_notebooks.py
│   ├── debug_resnet.py
│   ├── debug_probs.py
│   └── debug_gradcam.py
│
└── tests/                    # Pytest suite (existing)
    ├── test_data.py
    ├── test_eval.py
    ├── test_models_gradcam.py
    └── test_tb.py
```

---

## 12. How to Install Dependencies

### Local (venv)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Google Colab (recommended)
```python
# Run at top of notebook
!pip install -q -r requirements.txt
# Or install manually:
!pip install -q tensorflow scikit-learn matplotlib seaborn pillow nbformat
```

**Core dependencies:**
- `tensorflow` (>=2.13) — Keras 3, models, training, TensorBoard
- `numpy` — array operations
- `pandas` — experiment tables
- `matplotlib` — plotting
- `scikit-learn` — metrics (precision, recall, F1, ROC-AUC, confusion matrix)
- `seaborn` — optional, prettier confusion matrices
- `pillow` — image saving
- `nbformat` — notebook validation (for `scripts/build_notebooks.py`)

---

## 13. How to Run the Notebooks

### Option A: Google Colab (recommended)
1. Open `notebooks/member_X/model_training.ipynb` in Colab
2. Mount Google Drive when prompted (for data cache + TensorBoard logs)
3. Run cells sequentially

### Option B: Local Jupyter
```bash
jupyter lab
# Navigate to notebooks/member_X/model_training.ipynb
```

### Key setup cells in every notebook:
```python
# 1. Clone repo (Colab)
!git clone https://github.com/your-username/malaria-cnn-project.git
%cd malaria-cnn-project

# 2. Install deps
!pip install -q -r requirements.txt

# 3. Get data (downloads once, caches to Drive)
from common.data import get_data
cell_images = get_data()

# 4. Build datasets (identical split for everyone)
from common.data import build_datasets
train_ds, val_ds, test_ds = build_datasets(cell_images, augment=True)
```

---

## 14. TensorBoard Instructions

### During training (in notebook):
```python
from common.tb import ExperimentLogger

exp = ExperimentLogger(
    model_name="custom_resnet",
    exp_id=1,
    description="baseline",
    hparams=dict(lr=1e-3, optimizer="adam", batch_size=32, epochs=30,
                 augmentation=True, dropout=0.3, l2=0.0, strategy="scratch"),
    logdir="/content/drive/MyDrive/malaria_project/logs/tensorboard"
)
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=exp.callbacks())
exp.log_final_metrics(metrics_dict)
```

### Viewing logs:
```bash
# Local
tensorboard --logdir logs/tensorboard

# Colab
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/malaria_project/logs/tensorboard
```

### Naming convention:
```
custom_resnet_exp_01_baseline
custom_resnet_exp_02_lr_1e-4
transfer_model_1_exp_01_baseline
transfer_model_1_exp_02_unfreeze_last20
custom_cnn_exp_01_baseline
...
```

---

## 15. Results Organization

| Folder | Contents |
|--------|----------|
| `results/metrics/` | Per-experiment CSV/JSON with all 6 metrics |
| `results/figures/` | Training curves, final 4-plot figure |
| `results/confusion_matrices/` | Normalized CM PNGs |
| `results/roc_curves/` | ROC curve PNGs |
| `results/error_analysis/` | Misclassified cell images + analysis |
| `results/gradcam/` | Heatmap overlays + 3-example report grids |
| `models/` | Best weights per model (.weights.h5) |

All results should be committed to repo (except large model weights — use Drive).

---

## 16. Report Organization

Single unified report at `report/report.md` with skeleton:

1. **Introduction**
2. **Literature Review** (critical, >5 refs)
3. **Methodology** (dataset, preprocessing, augmentation, 4 models, training, metrics, explainability)
4. **Results and Discussion** (per-model experiments, final eval, model comparison, error analysis, Grad-CAM, critical discussion)
5. **Conclusion**
6. **References**
7. **Appendix**

**Update continuously** — do not wait until the end.

---

## 17. Contribution / Workflow

1. **Sync with team** before starting new experiment
2. **Create experiment entry** in `experiments/<model>/EXP-XX_<description>.md`
3. **Run experiment** with TensorBoard logging
4. **Save results** to `results/` folders
5. **Update report** section for your model
6. **Commit** with message: `exp: custom_resnet exp02 lr_1e-4`
7. **Pull before push** — resolve conflicts in experiment logs, not notebooks

See [docs/workflow.md](docs/workflow.md) for detailed Git workflow.

---

## Quick Start Checklist

- [ ] Clone repo
- [ ] Install requirements (`pip install -r requirements.txt`)
- [ ] Choose Transfer Model 1 & 2 (team decision)
- [ ] Assign models to members (update `src/utils/project_config.py`)
- [ ] Each member opens their `notebooks/member_X/model_training.ipynb`
- [ ] Run data setup cells (identical split guaranteed)
- [ ] Implement baseline model (EXP-01)
- [ ] Log to TensorBoard
- [ ] Document experiment
- [ ] Iterate through 7+ experiments
- [ ] Generate final evaluation figures
- [ ] Run Grad-CAM
- [ ] Write report section
- [ ] Prepare oral defense slides
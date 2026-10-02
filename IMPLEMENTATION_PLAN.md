# 🦟 Malaria CNN Classification Project — Implementation Plan (v3)

> **Goal:** Design, train, and evaluate 3 CNN models (custom ResNet + 2 transfer-learning) for automated malaria diagnosis from microscopic cell images, meeting every requirement in the FORMATIVE 2 rubric (35 pts total).
>
> **Status:** ✅ Dataset-access notebook received & analyzed → data pipeline finalized below.

---

## 0. Dataset Notebook — What It Tells Us

The official starter notebook (`Malaria-Diagnosis_CNN_Group6-EvenNumber.ipynb`) confirms:

| Fact | Detail | Impact on our plan |
|------|--------|--------------------|
| Data source | Direct NIH download: `https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip` — no auth needed | One shared download cell in every notebook; no gdrive file-ID workaround required |
| Data layout | Unzips to `cell_images/Parasitized/` + `cell_images/Uninfected/` (folder-per-class) | Use `tf.keras.utils.image_dataset_from_directory` or `image_dataset_from_directory`-style loading with label inference |
| Environment | Google Colab, GPU accelerator, Drive mounted at `/content/drive/` | All logs/checkpoints/artifacts must live under a Drive path to survive runtime disconnects |
| Persistence trap | `downloadData = True` re-downloads every session | Cache the unzipped data in Drive on first run; subsequent sessions just unzip from Drive (or download once + symlink) |
| Starter baseline | `Conv2D → MaxPool → Flatten → Dense → Dense` (Sequential API) | This is the *demo* baseline — our 3 models replace it; ResNet must NOT use Sequential (rubric requires Subclassing API) |
| References seeded | Rajaraman et al. 2018, Brownlee 2019, NIH + Bangalore Hospital acknowledgement | Reuse these as report references; add He et al. 2016, Selvaraju et al. 2017, Litjens et al. 2017, Yosinski et al. 2014, WHO 2023 |

**Dataset facts (NIH/NLM set):** 27,558 RGB cell images, ~13,779 per class (balanced), variable small sizes (~xx–1xx px wide, thin blood smear crops) → we resize to a fixed input size.

---

## 1. Understanding the Assignment

### What We're Building (3-person group)

| # | Model | Type | Target (held-out test set) |
|---|-------|------|---------------------------|
| 1 | **Custom ResNet** (hand-built, Keras Subclassing API) | From scratch | ≥90% sensitivity, ≥90% specificity |
| 2 | **Transfer Learning Model 1** (e.g., MobileNetV2) | Fine-tuned | ≥95% sensitivity, ≥90% specificity |
| 3 | **Transfer Learning Model 2** (e.g., DenseNet121 or EfficientNetB0) | Fine-tuned | ≥95% sensitivity, ≥90% specificity |

**Dataset:** NIH/NLM malaria cell images — 27,558 color images, ~50/50 parasitized/uninfected (balanced).

### The 6 Rubric Criteria (35 pts)

| Criterion | Pts | Key to full marks |
|-----------|-----|-------------------|
| Problem Framing & Literature Review | 5 | >5 scholarly refs, **critical** (not descriptive) review, each member links their model to literature |
| Model Implementation & Experimentation | 5 | 7+ well-documented experiments per model, TensorBoard evidence with run comparisons, justified choices |
| Evaluation, Results & Visual Analysis | 5 | Results tables, 4 plots per model, Grad-CAM incl. incorrect predictions, **critical interpretation** |
| Code Quality & Documentation | 5 | Modular, reproducible, Markdown cells explaining what/why/how-to-interpret |
| Report Quality & Collaboration | 10 | Scholarly prose (minimal bullets), correct citations, <20% AI text, integrated narrative |
| Presentation / Demo Video | 5 | Justify choices, interpret visuals live, answer follow-ups confidently |

**⚠️ Top scoring traps to avoid:**
1. Descriptive-only literature review (needs *critical comparison* + gaps)
2. Experiments that are reruns instead of *meaningful variations*
3. Grad-CAM shown but only *described*, not *analyzed* (and missing the incorrect-prediction case)
4. Bullet-point-heavy report instead of academic prose
5. TensorBoard run names like `test1`, `final` — use descriptive names
6. Report claims not traceable to a logged run

---

## 2. Deliverables Checklist (from rubric)

| Item | Requirement | How We'll Meet It |
|------|-------------|-------------------|
| Models | 3 for 3-person group | Custom ResNet + MobileNetV2 + DenseNet121/EfficientNetB0 |
| Experiments | 7+ per model (21+ total) | Campaign per model in §5 |
| Metrics table | Acc/Prec/Rec/F1/ROC-AUC per experiment | Shared `evaluate_model()` helper → consistent tables |
| Visualizations (final model) | 4 plots, labeled & interpreted | Shared plotting helpers, single-row 1×4 layout per model |
| Explainability | Grad-CAM on 3 example types per model | Shared Grad-CAM utils handling subclassed models |
| Error analysis | Misclassifications + overfit/underfit diagnosis | Shared misclassification viewer + curve diagnosis |
| Group report | 10–15 pages, <20% AI text | Report outline in §8, scholarly prose, member-written, Human Touch guide §10 |
| Notebook | 1 per member, no shared template | 3 notebooks, shared `common/` module imported (not copied) |
| Contribution sheet | 1 per group | Fill from ownership map in §5 |
| TensorBoard evidence | Screenshots/exports + Drive link to raw logs | Log to Drive-backed `logs/fit/`, screenshot comparisons |
| Oral defense | Individual 10–15 min slots | Prep answers for §9 question bank + §10 ownership practice |

---

## 3. Repo Structure

```
Malaria-CNN-Classification-Project-/
├── IMPLEMENTATION_PLAN.md          # this file
├── Malaria-Diagnosis_CNN_Group6-EvenNumber.ipynb   # official dataset-access starter (reference)
├── common/
│   ├── data.py                     # download + cache, splits, augmentation pipeline
│   ├── eval.py                     # metrics table, 4 required plots, error-analysis helpers
│   ├── tb.py                       # TensorBoard run dirs, callbacks, hparams logging
│   └── gradcam.py                  # Grad-CAM (incl. subclassed-model support)
├── notebooks/
│   ├── member1_custom_resnet.ipynb
│   ├── member2_mobilenetv2.ipynb
│   └── member3_densenet121.ipynb
├── logs/fit/                       # TensorBoard event logs (Drive-backed, zipped per session)
├── artifacts/
│   ├── checkpoints/
│   ├── final_plots/                # per-model 4-plot figures
│   └── gradcam/                    # heatmap figures
├── report/                         # report draft + figures
└── README.md
```

**Important:** rubric says "no shared template" for notebooks. The clean way to comply: each member's notebook imports `common/` utilities but has **their own** Markdown prose, their own experiment decisions, and their own narrative voice.

---

## 4. Data Pipeline (finalized from the notebook)

### 4.1 Download & Cache Strategy

```python
# common/data.py — cell 1 of every notebook
DATA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
DRIVE_DATA_DIR = "/content/drive/MyDrive/malaria_project/data"   # persistent cache
LOCAL_DATA_DIR = "/content/cell_images"

def get_data():
    # 1. If Drive cache has cell_images.zip → copy to local + unzip (fast, no re-download)
    # 2. Else → wget once, then ALSO copy the zip into Drive cache for next sessions
    # 3. Verify: count PNGs in Parasitized/ and Uninfected/ (expect ~13,779 each)
```

**Why:** the starter notebook re-downloads ~700MB every Colab session (`downloadData=True`). Downloading once and caching the zip in Drive saves hours across 21+ experiment runs.

### 4.2 Loading & Splitting

```python
# stratified 70/15/15 train/val/test — single split shared by ALL members
# (comparable metrics require identical test sets)
def build_datasets(img_size, batch_size, augment=False):
    # tf.keras.utils.image_dataset_from_directory(
    #     LOCAL_DATA_DIR, labels="inferred", label_mode="binary",
    #     color_mode="rgb", batch_size=batch_size, image_size=img_size,
    #     shuffle=True, seed=42, validation_split=0.30, subset="both")
    # Then split the 30% validation portion 50/50 → val 15% / test 15%
```

**Decisions locked in:**
- **Label mapping:** 1 = Parasitized (infected, positive class), 0 = Uninfected → recall = sensitivity (infected), specificity = TNR on uninfected. Document in every notebook header.
- **Split discipline:** ALL members use the same split (same `seed=42`) so metrics are comparable across models and the report can compare fairly.
- **Input size: 128×128** as default for speed (21+ runs); DenseNet121 gets its own resolution experiment (128 vs 224). MobileNetV2 accepts ≥32; ResNet50/EfficientNetB0/DenseNet121 prefer 224 — document any resize.
- **Normalization:** use a `Rescaling(1./255)` layer inside the model (or `preprocess_input` per backbone — for MobileNetV2/EfficientNet/DenseNet use their `tf.keras.applications.*.preprocess_input`, and match it consistently).

### 4.3 Augmentation (on/off per experiment)

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])  # cells are roundish; flips/rotations are label-preserving — cite this in report
```

---

## 5. Model Designs & Experiment Campaigns

### Model 1 — Custom ResNet (Subclassing API, from scratch)
**Paper:** He et al., 2016 — *Deep Residual Learning for Image Recognition*

**Design (SmallResNet-14-ish for 128×128 input):**
- Stem: `Conv 3×3 /32 → BN → ReLU`
- 3 residual stages: 64 → 128 (stride 2) → 256 (stride 2)
- BasicBlock ×2 per stage: `Conv3×3 → BN → ReLU → Conv3×3 → BN` + identity/projection shortcut → ReLU
- Projection shortcut (1×1 conv) when shape changes
- GlobalAveragePooling → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)

**Experiment campaign (7+):**

| # | Run name | Change | Rationale |
|---|----------|--------|-----------|
| 01 | `custom_resnet_exp_01_baseline` | LR 1e-3, Adam, batch 32, no aug, no dropout | Reference floor |
| 02 | `custom_resnet_exp_02_bn_variants` | BN placement/epsilon tweaks | BN is the key from-scratch trick (Ioffe & Szegedy, 2015) |
| 03 | `custom_resnet_exp_03_wider_deeper` | Wider stages (32→64→128) or +1 block/stage | Capacity increase |
| 04 | `custom_resnet_exp_04_augmentation` | + flips/rotations/zoom (0.15) | ~27k images — augmentation is the main generalization lever |
| 05 | `custom_resnet_exp_05_l2` | + L2 1e-4 | Weight decay vs dropout for conv nets |
| 06 | `custom_resnet_exp_06_lr_schedule` | Cosine decay or ExponentialDecay | Smooth convergence |
| 07 | `custom_resnet_exp_07_final` | Best combo, longer training | Final submission model |
| 08 | `custom_resnet_exp_08_depth_ablation` | +1 stage (spare) | Only if time allows |

**Subclassing skeleton (rubric-critical):**

```python
class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same",
                                   kernel_regularizer=regularizers.l2(l2))
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding="same",
                                   kernel_regularizer=regularizers.l2(l2))
        self.bn2   = layers.BatchNormalization()
        self.proj  = None
        if stride != 1:                       # projection shortcut (He et al. §3.2)
            self.proj = keras.Sequential([
                layers.Conv2D(filters, 1, strides=stride),
                layers.BatchNormalization()])

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        shortcut = self.proj(x, training=training) if self.proj else x
        return tf.nn.relu(out + shortcut)

class CustomResNet(keras.Model):
    def __init__(self, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stem = keras.Sequential([layers.Rescaling(1./255),
            layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization()])
        self.stage1 = [ResidualBlock(64),  ResidualBlock(64)]
        self.stage2 = [ResidualBlock(128, stride=2), ResidualBlock(128)]
        self.stage3 = [ResidualBlock(256, stride=2), ResidualBlock(256)]
        self.gap   = layers.GlobalAveragePooling2D()
        self.head  = keras.Sequential([layers.Dense(256), layers.Dropout(0.3),
                                       layers.Dense(1, activation="sigmoid")])

    def call(self, x, training=False):
        x = tf.nn.relu(self.stem(x, training=training))
        for blk in self.stage1 + self.stage2 + self.stage3:
            x = blk(x, training=training)
        return self.head(self.gap(x), training=training)
```

⚠️ **Two subclassing gotchas to handle explicitly:**
1. **Grad-CAM access (assignment §5.4):** the above structure lets us build a `feature_tap` sub-model by keeping block objects as named attributes — we can reconstruct `stem→stages→gap` as an intermediate `keras.Model` after calling it once on a dummy input. Test this in exp_01, before the campaign.
2. **Saving:** subclassed models need `get_config()`/`from_config()` implemented for `model.save()` — or simpler: save weights only (`model.save_weights("...weights.h5")`) and rebuild the object in the eval notebook.

### Model 2 — Transfer Learning 1: **MobileNetV2**
**Why:** lightweight (3.5M params) → fast iterations for 7+ experiments; different family from DenseNet (inverted residuals vs dense connectivity) → richer comparative analysis.

**Experiment campaign (7+):**

| # | Run name | Change | Rationale |
|---|----------|--------|-----------|
| 01 | `mobilenet_exp_01_frozen_baseline` | ImageNet weights, frozen base, head-only training | Standard 2-phase fine-tune |
| 02 | `mobilenet_exp_02_unfreeze_last20` | Unfreeze last 20 layers, LR 1e-5 | Lower LR needed when unfreezing |
| 03 | `mobilenet_exp_03_lr_comparison` | 1e-4 vs 1e-5 for fine-tune phase | Too-high LR destroys pretrained features |
| 04 | `mobilenet_exp_04_augmentation` | + augmentation | Generalization |
| 05 | `mobilenet_exp_05_dropout_l2` | Dropout 0.3 → 0.5, + L2 1e-4 | Regularization tuning |
| 06 | `mobilenet_exp_06_batch_size` | Batch 64 vs 32 | Throughput vs generalization |
| 07 | `mobilenet_exp_07_final` | Best combo | Final submission model |

### Model 3 — Transfer Learning 2: **DenseNet121 or EfficientNetB0**
**Why:** different family from MobileNetV2 (dense connectivity) → richer comparative analysis.

**Experiment campaign (7+):** same template as MobileNetV2, plus a **resolution experiment** (224 vs 128 input) since DenseNet121 expects 224×224:
- `densenet_exp_01_frozen_baseline` … `densenet_exp_07_final`

---

## 6. Team Ownership (3-person groups)

| Member | Model | Notebook | Report section |
|--------|-------|----------|----------------|
| Member 1 | Custom ResNet | `notebooks/member1_custom_resnet.ipynb` | Methodology (ResNet design) + Results 4.1 |
| Member 2 | MobileNetV2 | `notebooks/member2_mobilenetv2.ipynb` | Methodology (TL 1) + Results 4.2 |
| Member 3 | DenseNet121 | `notebooks/member3_densenet121.ipynb` | Methodology (TL 2) + Results 4.3 |
| All | — | — | Intro, Lit Review, Comparative Analysis, Conclusion (split) |

---

## 7. Shared Code Design (`common/`)

### Constants
```python
IMG_SIZE = (128, 128)        # revisit after EDA
BATCH_SIZE = 32
SEED = 42
LABEL_MAP = {"Parasitized": 1, "Uninfected": 0}   # 1 = infected (positive class)
CALLBACKS = ["checkpoint", "early_stop"]
FINAL_PLOTS_DIR = "artifacts/final_plots"
GRADCAM_DIR = "artifacts/gradcam"
```

### Shared evaluation helpers (`common/eval.py`, sklearn-based)
```python
- evaluate_model(model, test_ds, y_true) → dict: accuracy, precision, recall, f1, roc_auc, specificity
- plot_accuracy_curves(history)          # train vs val
- plot_loss_curves(history)
- plot_confusion_matrix(y_true, y_pred)  # labels ["Parasitized", "Uninfected"]
- plot_roc_curve(y_true, y_score)
- combined_row_of_4_plots(...)           # rubric wants single-row layout per model
```

### Shared TensorBoard setup (`common/tb.py`)
```python
- get_run_logdir(model_name, exp_id, description) → logs/fit/custom_resnet_exp_01_baseline
- build_callbacks(checkpoint_dir, log_dir) → ModelCheckpoint, EarlyStopping, TensorBoard
- Log hparams: lr, optimizer, batch size, epochs, augmentation, dropout, L2, strategy
- Also log final test metrics via tf.summary.scalar in the same run dir so the
  TensorBoard comparison view shows final evaluation metrics alongside curves.
```

### Shared Grad-CAM utilities (`common/gradcam.py`)
```python
- make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None)
- For subclassed ResNet: build an intermediate sub-model exposing the last conv block's output
- display_heatmap(img, heatmap, alpha=0.4, cmap="jet")
- pick_examples(y_true, y_pred, y_score) → (best_correct_infected, best_correct_uninfected, worst_incorrect)
- overlay_grid(images, heatmaps, titles)  # 1×3 figure for report
```

### Shared data pipeline (`common/data.py`)
```python
- get_data()                          # NIH URL download + Drive cache (from starter notebook)
- build_datasets(img_size, batch_size, augment=False)   # stratified 70/15/15, seed=42
- class distribution check + sample grid
- label mapping: 1=Parasitized, 0=Uninfected
```

**Key design decision for the subclassed ResNet:** build `CustomResNet(Model)` with a `stem → stages → head` structure where the last conv output can be tapped via a sub-model built from named layers — this avoids the assignment §5.4 trap of "fully encapsulated models hide conv outputs."

---

## 8. Report Outline (10–15 pages, scholarly prose)

1. **Introduction** — malaria burden in Sub-Saharan Africa (WHO 2023), microscopy limitations, AI-assisted diagnosis motivation
2. **Literature Review** — CNNs in medical imaging (Litjens et al., 2017 survey), ResNet (He et al., 2016), transfer learning (Yosinski et al., 2014), prior malaria CNN work (Rajaraman et al., 2018 — 96.99% sens / 97.75% spec), Grad-CAM (Selvaraju et al., 2017). **Must be critical: compare approaches, identify gaps.**
3. **Methodology** — dataset (NIH/NLM 27,558 images), preprocessing pipeline, custom ResNet design rationale, both transfer-learning models with justification
4. **Results & Discussion**
   - 4.1 Experiments (one table per model, run names = TensorBoard names)
   - 4.2 Evaluation visualizations (4 plots in a single row per model + interpretation)
   - 4.3 Comparative analysis of the three models
   - 4.4 Error analysis per model (2–3 misclassified examples + overfit/underfit diagnosis citing specific curve evidence)
   - 4.5 Explainability subsection (custom-vs-pretrained differences, incorrect-prediction attention, confidence impact)
5. **Critical Discussion** — successes, failures, limitations
6. **Conclusion**
7. **References** (IEEE or APA, consistent, ≥5 credible scholarly sources)
8. **Appendix** — Colab notebook links + open-access Google Drive link to raw TensorBoard logs

---

## 9. Oral Defense Prep (per member, ~10–15 min)

**2-min overview:** architecture, key design choices (residual connections / transfer learning, augmentation, regularization, fine-tuning strategy).

**Q&A question bank — prepare answers for:**
- What is the model looking at when it makes this prediction?
- Why do you believe that region matters?
- What happened when the model was wrong?
- Can Grad-CAM prove the model learned medically correct features — why or why not? *(No — attention ≠ causal proof; heatmaps are evidence to question behavior, not proof of correct reasoning.)*
- Why this optimizer / LR / batch size? What did the experiments show?
- Where does your model rank among the group's 3, justified by experimental evidence not just accuracy?

*(Pair this prep with the Human Touch guide in §10 — the defense is where ownership is ultimately proven.)*

---

## 10. Human Touch — Authentic Voice & Ownership Guide

The rubric scores human authorship directly: the report criterion demands **<20% AI-generated text** and visible *student voice*; the defense demands members explain decisions with genuine ownership. This section is our playbook for keeping every artifact authentically ours — not a bolt-on compliance step.

### 10.1 The golden rule
**AI is a brainstorming partner, never a ghostwriter.** Allowed: explaining concepts, suggesting structure, debugging code, understanding error messages. Not allowed: generating prose that lands in the report or notebooks. Every paragraph a marker reads must have been written by a member, in their own words, from their own experiment history.

### 10.2 Write from the experiment journey — keep a lab journal
- From day one (M0), keep a shared **decision log** (one markdown file in `report/journal.md`). After every experiment, each owner jots four lines: *what I changed, what I expected, what actually happened, what surprised me / what's next.*
- The journal is the raw material for authentic writing. "Our first attempt overfit badly — training accuracy hit 99% by epoch 6 while validation stalled at 94% — so we added augmentation" is a human sentence no AI can invent, because it comes from *our* run history.
- **Honest failures are gold.** Dead-end experiments, the bug that cost an afternoon, the wrong hypothesis — markers read these as evidence of real work, and the rubric explicitly rewards *documented progression*, not just wins. Don't clean the story up; tell it.

### 10.3 Concrete details are the fingerprint of human authorship
- Cite *our actual numbers*: "validation loss plateaued near epoch 9 at 0.28", "`mobilenet_exp_03_lr_comparison` diverged once we unfroze at 1e-4".
- Reference our own figures by number and interpret them: "Figure 4 shows the train–validation gap widening from epoch 12, which is why we added L2 in the next run."
- Generic, could-have-been-written-by-anyone text is the fingerprint of AI. Specificity — run names, epochs, loss values, misclassified-image descriptions — is our protection.

### 10.4 Voice rules for the report
- **First person plural** for group sections ("we trained", "our split"); **first person singular** inside each member's model section ("I chose MobileNetV2 because…").
- **Hedge like a scientist:** "this suggests", "a plausible explanation is", "we suspect". Absolute overconfidence reads as machine-generated.
- **Prose over bullets** — the rubric explicitly penalizes bullet-heavy reports. Tables carry the numbers; paragraphs carry the meaning.
- **Sentence rhythm:** mix short and long. Read every paragraph aloud — if it doesn't sound like something you'd actually say, rewrite it.
- **Ban list (AI-tell vocabulary):** *delve, crucial, leverage, showcase, robust, seamless, pivotal, "it is important to note", "in the realm of", "furthermore" chained across paragraphs.* If the sentence works without them, cut them.
- **Local grounding:** anchor the introduction in *our own region's* malaria reality — East African / national NMCP reports alongside WHO African-region figures. It's a detail only a team from here would naturally include, and it strengthens Problem Framing (criterion 1).

### 10.5 Notebooks that smell like their author
- Each member writes their **own Markdown cells in their own voice** — explaining the *why*, including what went wrong mid-way and how they noticed.
- Sharing the `common/` module is fine (and good engineering); **identical phrasing across notebooks is not** — that's the "shared template" red flag the rubric warns about.
- Keep the messy middle visible: the cell where the first architecture failed, with a comment about what it taught us, followed by the fixed version. Progression is the story.

### 10.6 The defense is the ultimate AI detector
- The Q&A will probe whether we can **re-derive our own decisions**: why this learning rate, why this depth, what happened when the model was wrong. The only reliable way to get there is to actually run the experiments ourselves and keep the journal current.
- **Practice ritual (twice before the defense):** each member explains their model to another member in 5 minutes without slides; the listener asks hostile questions ("why not just accuracy?", "prove that region matters").
- If AI helped you *understand* a concept, the bar is: can you re-explain it from scratch on a whiteboard? That's the standard the Q&A sets — and the reason understanding, not output, is what we optimize for.

### 10.7 Integrity guardrails
- Paraphrase from sources, always with citation; zero copy-paste from papers, blogs, or other notebooks.
- Before submission: one **read-aloud pass per member** on their own sections, plus an AI-detector scan as a noisy smoke test. The real protection is that the text is genuinely ours — detectors are unreliable, but our writing shouldn't need one.
- Target ~0% AI in final text, not merely <20%. Anything AI-assisted stays at the level of *ideas and structure*, never sentences.

---

## 11. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 21+ runs × re-downloading 700MB every session | High | **Drive cache (§4.1):** download NIH zip once, reuse thereafter |
| TensorBoard logs lost on Colab disconnect | High | Mount Drive; write `logs/fit` + checkpoints to Drive-backed paths |
| Custom ResNet fails 90/90 target | Medium | Baseline first without augmentation, BN tuning early (Exp 02), widening/deepening; start early |
| Grad-CAM fails on subclassed model | Medium | Named block attributes + intermediate sub-model; **smoke-test in exp_01** before the campaign |
| Subclassed model checkpointing | Low | Save weights only (`.weights.h5`); rebuild object for eval |
| Class mapping confusion | Low | 1=parasitized, 0=uninfected, documented everywhere |
| Notebooks look "shared template" | Medium | Distinct Markdown prose, distinct experiment ordering/choices, shared code imported not duplicated |
| Report flagged >20% AI | High | **Human Touch guide (§10):** journal-first writing, member-written prose, ban list, read-aloud pass — target ~0% AI |
| Different members using different splits → incomparable metrics | Medium | One shared split (seed=42) in `common/data.py`, used by all 3 notebooks |

---

## 12. Milestones

| Milestone | Owner(s) | Output |
|-----------|----------|--------|
| M0 — Shared foundation | All | `common/` utilities committed, data cached in Drive, split verified identical across notebooks |
| M1 — Baselines (Exp 01) | All | All 3 models run end-to-end once; **Grad-CAM smoke-tested on subclassed ResNet** |
| M2 — Experiment campaigns | Individual owners | 7+ runs per model, TensorBoard comparison screenshots captured |
| M3 — Final model + evaluation pack | Individual owners | Final 4 plots + metrics table per model |
| M4 — Explainability | All | Grad-CAM examples (3 per model), interpretations drafted |
| M5 — Error analysis | All | Misclassified examples documented with plausible causes |
| M6 — Report + defense prep | All | 10–15 page report, defense Q&A prep done |

---

## 13. Definition of Done (per rubric)

### Models & Experiments
- [ ] 3 models: 1 custom ResNet (Subclassing API, explicit `call()`) + 2 different pretrained architectures
- [ ] 7+ documented experiments per model (21+ total), each with meaningful variation (not reruns)
- [ ] Every experiment logged to TensorBoard with descriptive run name following convention
- [ ] TensorBoard comparison screenshots/exports captured as evidence
- [ ] Raw TensorBoard logs uploaded to Google Drive with open-access link in report Appendix

### Metrics & Evaluation
- [ ] Per-experiment metrics table: Accuracy, Precision, Recall, F1, ROC-AUC (one table per model)
- [ ] Metrics reported under both names: recall = sensitivity, specificity = TNR on uninfected
- [ ] All 4 required plots for each model's final version, labeled and interpreted
- [ ] ROC-AUC appears in both the metrics table AND as a full curve
- [ ] Grad-CAM per model: 1 correct-infected + 1 correct-uninfected + 1 incorrect, with interpretation

### Report & Documentation
- [ ] 10–15 pages, scholarly prose (minimal bullets), ≥5 credible scholarly references
- [ ] All sections present: Intro, Lit Review, Methodology, Results & Discussion, Conclusion, References, Appendix
- [ ] Experiments table per model with run names matching TensorBoard logs exactly
- [ ] Error analysis per model: 2–3 misclassified examples + overfit/underfit diagnosis citing curve evidence
- [ ] Comparative analysis across all 3 models, justified by experimental evidence
- [ ] Explainability subsection: custom-vs-pretrained differences, incorrect-prediction attention, confidence impact
- [ ] <20% AI-generated text (members write in their own words)
- [ ] Lab journal / decision log kept from M0 — the source of authentic detail
- [ ] Each member wrote their own sections in their own words; read-aloud voice pass done per member
- [ ] No AI-tell phrasing (ban list §10.4); prose-first, bullets only for factual lists
- [ ] Report cites our actual run names, numbers, and figure numbers throughout

### Code & Notebooks
- [ ] 1 notebook per member (no shared template)
- [ ] Modular, reproducible code: functions, constants block, seeds
- [ ] Markdown cells explaining what/why/how-to-interpret for every code block
- [ ] Consistent style across notebooks

### Presentation
- [ ] Each member: 2-min overview + Q&A prep done
- [ ] Ready for the question bank in §9
- [ ] Ranking among the group's 3 models stated with evidence

---

## 14. First Session Checklist (when execution starts)

1. ✅ Confirm GPU is active in Colab (`tf.test.gpu_device_name()` → `/device:GPU:0`)
2. Download NIH zip once → cache in Drive (`/content/drive/MyDrive/malaria_project/data/`)
3. Verify counts: ~13,779 Parasitized + ~13,779 Uninfected PNGs
4. EDA: sample grid, image size distribution → confirm 128×128 choice
5. Build the shared 70/15/15 split (seed=42) and verify identical across all 3 notebooks
6. Create `report/journal.md` decision log — every experiment gets four lines from its owner (§10.2)
7. Train ResNet exp_01 baseline → confirm loss decreases, sanity metrics
8. **Immediately smoke-test Grad-CAM on the subclassed ResNet** (assignment §5.4 trap)
9. Then launch the full experiment campaigns

---

*Last updated: 2026-09-30 — v3: added Human Touch guide (§10) covering authentic voice, journal-driven writing, defense ownership, and integrity guardrails; later sections renumbered.*

# MEMBER 1 — MobileNet Transfer Learning — Implementation Plan

> **Scope lock (absolute):** I am MEMBER 1, GROUP LEADER. My model is **MobileNet
> Transfer Learning** (= `transfer_model_1` in the group structure). All work below
> touches ONLY: `member_1/`, `experiments/transfer_model_1/`,
> `logs/tensorboard/transfer_model_1/`, my report contribution, and shared files
> I *reuse read-only* (`common/`, `src/utils/project_config.py` — one allowed
> registration edit, see §2.1).
> Members 2/3/4 (custom ResNet, other transfer model, additional custom CNN) are
> out of scope. No shared model implementation. No assumed or fake results.

---

## 0. What I understood (operating rules)

1. **One model, one owner.** MobileNet transfer learning, binary malaria
   classification, ImageNet pretrained, frozen → fine-tuned.
2. **7+ meaningful experiments**, controlled one-variable changes, logical
   progression, failures KEPT as evidence, no reruns-for-show.
3. **TensorBoard for every run** with descriptive names
   (`mobilenet_exp_01_baseline`), full hparams (lr, optimizer, batch size,
   epochs, augmentation, dropout, L2, frozen/unfrozen strategy), train/val
   curves, and final metrics written into the SAME run. Real metrics only —
   never typed in.
4. **Metrics:** accuracy, precision, recall, F1, ROC-AUC per experiment;
   sensitivity (= recall on Parasitized) + specificity (= TNR on Uninfected)
   explicit and consistent; class convention Parasitized=1 / Uninfected=0.
5. **Leakage discipline:** test set untouched during selection — experiments
   are guided by validation; ONE final test evaluation after selection.
6. **Final model** selected by multi-metric rationale (not accuracy alone);
   targets ≥95% sensitivity / ≥90% specificity to report and discuss.
7. **Final artifacts:** 4 figures (acc curve, loss curve, confusion matrix,
   ROC/AUC), error analysis (real predictions, observation vs possible
   explanation), Grad-CAM (final conv layer verified, logit-based, 3 example
   types: correct-infected / correct-uninfected / incorrect), interpreted
   critically — never claimed as proof of medical reasoning.
8. **Report contribution:** clean draft/outline in MY section only, with
   `[FILL]` markers for my own interpretation and `[REFERENCE NEEDED]` instead
   of any invented citation. I rewrite narrative into my own voice.
9. **Code quality:** minimal, modular, reproducible, no dead code, no fake
   data, seeded, config-driven. Optimized for *explainability at the defense*,
   not impressiveness.
10. **Workflow:** inspect → report gaps → smallest implementation for EXP01 →
    verify → decide next experiment from actual results → repeat to 7+ →
    select final → final eval → visuals/error/Grad-CAM → report draft → audit.

---

## 1. Repository inspection findings (spec §2 checklist)

| # | Question | Finding | Verdict |
|---|----------|---------|---------|
| 1 | Repo structure | Group skeleton exists: `member_1..4/`, `src/`, `docs/`, `report/`, `results/`, `experiments/{custom_resnet,custom_cnn,transfer_model_1,transfer_model_2}/`, `logs/tensorboard/{...}` | ✔ unambiguous |
| 2 | Member 1 workspace | `member_1/model_training.ipynb` (15-cell starter: simple Sequential CNN — NOT MobileNet yet) + `experiments/transfer_model_1/` (empty) + `logs/tensorboard/transfer_model_1/` | ✔ my lane |
| 3 | Dataset utilities | `common/data.py`: NIH download + Drive cache, exact 70/15/15 split by file (seed 42), label flip **Parasitized=1**, split fingerprints; 45 passing tests verify all of it | ♻ **REUSE — do not duplicate** |
| 4 | Env/deps | `.venv` (local, TF 2.21 CPU — smoke tests only), `requirements.txt`; real training runs on Colab GPU | ✔ |
| 5 | Existing MobileNet code | `common/models.py::build_transfer_model("mobilenetv2", ...)` + `unfreeze_top_n` — tested; backbone preprocessing lives inside the model graph (no double-preprocess bug) | ♻ **REUSE** |
| 6 | TensorBoard | `common/tb.py::ExperimentLogger` — enforces rubric naming, logs hparams + CSV + final metrics into the run dir; `zip_logs_for_drive` for evidence export | ♻ **REUSE** |
| 7 | Evaluation utils | `common/eval.py`: acc/precision/recall/F1/ROC-AUC/**specificity**, sensitivity alias, confusion matrix, ROC, 4-in-1 final figure, misclassification mining — display-label bug found & fixed & regression-tested | ♻ **REUSE** |
| 8 | Plotting utils | same module + `matplotlib` (headless-verified) | ♻ **REUSE** |
| 9 | Grad-CAM | `common/gradcam.py`: eager GradientTape, **logit-based** (sigmoid saturation kills heatmaps — tested), works on functional backbones incl. MobileNetV2, `pick_gradcam_examples` returns the 3 required example types | ♻ **REUSE** |
| 10 | My starter notebook | `member_1/model_training.ipynb` — structure outdated (Sequential CNN, no MobileNet, no campaign) | ✏ **REBUILD IN PLACE** (preserve nothing broken; it is my file) |

**Gaps (what does not exist yet):** the MobileNet architecture in my notebook,
the 7-experiment campaign, per-experiment validation-metric logging, final
selection logic, final artifacts, Grad-CAM section, comparison table, report
contribution draft.

---

## 2. Discrepancies found — decisions needed before coding (spec §24)

### 2.1 `src/utils/project_config.py` says `transfer_model_1: owner=TBD, architecture=TBD`
The file's own docstring says "Update this file when team decisions are made."
Registering MY decision is the intended use — one additive edit:
`owner="Member 1"`, `architecture="MobileNetV2"`. No other fields touched.
*(Will apply after you confirm MobileNetV2.)*

### 2.2 Run-name prefix: your spec vs the group config
Your spec §8 mandates `mobilenet_exp_01_baseline`. The group config's generic
convention would produce `transfer_model_1_exp_01_baseline`. **Plan: use
`mobilenet_*` run names** (your explicit instruction; TensorBoard names must
match the report table I write). Run dirs still live under the group slot
`logs/tensorboard/transfer_model_1/`. → confirm below.

### 2.3 `docs/team_responsibilities.md` assigns Member 1 "Data Acquisition and Preparation", not a model
That doc is an unfilled placeholder (`[Name]` everywhere). Your instruction is
authoritative: Member 1 = MobileNet. I will not edit team docs (out of my
scope) — flag for the group to fix.

### 2.4 Earlier group-wide scaffolding from the previous session
`common/` + `tests/` (45 green tests) = **shared, verified infrastructure I
reuse read-only**. `notebooks/01_custom_resnet…` and `03_densenet…` target
other members' models — **I stop working on them** (left untouched; teammates
may adopt or discard). `notebooks/02_mobilenetv2…` and the old
`IMPLEMENTATION_PLAN.md` are superseded by this plan + my rebuilt notebook.
→ confirm whether to keep or remove the stale drafts (I recommend: leave
everything in place, zero destructive changes).

---

## 3. Architecture (spec §4) — simple and defensible

```
Input (128×128×3, raw 0–255 RGB)
  ↓  MobileNetV2 preprocess_input        ← inside the model graph (no leaks)
  ↓  MobileNetV2 backbone (ImageNet)     ← frozen in exp 01–04; top-N unfrozen later
  ↓  GlobalAveragePooling2D
  ↓  Dropout(0.3)
  ↓  Dense(1, sigmoid)                   ← P(Parasitized)
```

- Binary sigmoid + `binary_crossentropy`; balanced classes (13,779 vs 13,779).
- **Phase 1 (exp 01–04):** backbone frozen — train head only. Fast, stable,
  clean baseline for controlled variable studies.
- **Phase 2 (exp 05–07):** unfreeze controlled top-N of the backbone at a
  10× lower LR (discriminative fine-tuning), before-compile `unfreeze_top_n`.
- No extra layers, no fancy tricks — every block must be explainable in the
  oral defense in one sentence.

## 4. Data & leakage discipline (spec §5, §11)

- Reuse `common/data.py::get_data` (Drive-cached, no re-download per session)
  and `build_datasets(seed=42)`: exact **70/15/15 split by file** → identical
  split for all members (fingerprint printed in notebook §3/§5 as proof).
- Augmentation (flip/rot/zoom) applied **train-only** via
  `build_datasets(augment=True)` — val/test never augmented (tested).
- **Experiments 01–07 report VALIDATION metrics** (guidance + model selection).
- **Test set evaluated exactly once**, after final selection (§19 of notebook).
- Fingerprints + `EXPECTED_TOTAL=27,558` sanity check printed; no invented
  statistics — the notebook prints whatever the data says.

## 5. Experiment campaign (spec §6–§8)

Controlled one-variable changes; each hypothesis written BEFORE the run; each
result logged on validation; poor results kept.

| run name | variable | hypothesis |
|---|---|---|
| `mobilenet_exp_01_baseline` | — reference | frozen backbone, Adam 1e-3, bs 32, 30 epochs, dropout 0.3: establish reference behaviour |
| `mobilenet_exp_02_lr_1e-4` | LR 1e-3→1e-4 | gentler head convergence; stability reference for fine-tuning |
| `mobilenet_exp_03_augmentation` | + flip/rot/zoom | augmentation should close the train/val gap before touching backbone |
| `mobilenet_exp_04_dropout_0.5` | dropout 0.3→0.5 | regularize the head if exp 03 shows residual overfit |
| `mobilenet_exp_05_unfreeze_last20` | top-20 layers trainable @ 1e-4 | discriminative fine-tuning buys task features cheaply |
| `mobilenet_exp_06_finetune_lr_1e-5` | fine-tune @ 1e-5 | lower LR protects pretrained features during fine-tuning |
| `mobilenet_exp_07_final_refinement` | best combination | the recipe we select as final (declared by evidence) |

**EXP 08+ rule:** only if evidence justifies (e.g. exp 05 overfits → try
unfreeze_last10; or stable → try unfreeze_last40). Never forced. Failures stay
in the table with their outcome noted.

**Decision protocol after each run:** compare validation acc/ROC-AUC/sensitivity
vs previous best → write one logbook paragraph → choose next variable.

## 6. TensorBoard & metrics per run (spec §9–§10)

Per run, `ExperimentLogger` writes into
`logs/tensorboard/transfer_model_1/<run_name>/`:

- **HPARAMS record:** lr, optimizer, batch_size, epochs, augmentation, dropout,
  L2, strategy (frozen_head / unfreeze_lastN), model=MobileNetV2 → hparams
  dashboard comparison screenshots for the report.
- **Scalars:** epoch train/val loss + accuracy curves.
- **CSV** `training_log.csv` (evidence that survives TB issues).
- **Final metrics** (computed by `evaluate_model` on the *validation* set):
  accuracy, precision, recall, F1, ROC-AUC, specificity, sensitivity →
  `final_metrics.txt` + scalars in the SAME run dir.
- Naming enforced by `make_run_name` → exactly the §5 table names.
- `zip_logs_for_drive` → raw event dirs preserved to Drive for the appendix.

## 7. Final model selection (spec §12)

1. Build the comparison table from logged runs (validation metrics).
2. Rank by **sensitivity first (clinical priority), then specificity + ROC-AUC
   + F1** — never accuracy alone; state the trade-off explicitly.
3. Check against targets: ≥95% sensitivity, ≥90% specificity (report & discuss,
   not a gate).
4. **One** final test evaluation of the selected config, written to its run dir
   and clearly labeled FINAL TEST.

## 8. Final artifacts (spec §13–§15)

- 4 figures via `common/eval.py` (single-row report figure + individual PNGs
  into `results/figures/`): accuracy curve, loss curve, confusion matrix
  (labeled 1=Parasitized), ROC/AUC.
- **Error analysis:** `find_misclassifications` (most-confident errors) → 2–3
  examples with image, true/pred, confidence, OBSERVATION vs POSSIBLE
  EXPLANATION kept distinct; curve diagnosis (over/underfitting) tied to the
  specific experiments that addressed each symptom.
- **Grad-CAM:** last conv layer of MobileNetV2 identified by name; logit-based
  heatmaps (`make_gradcam_heatmap`); the 3 required example types via
  `pick_gradcam_examples`; grid + per-example overlays into
  `results/gradcam/`; interpretation prompts with `[FILL]` for my own words;
  explicit limitation note (Grad-CAM shows *where*, not *whether medically
  valid*).

## 9. Notebook structure (spec §18) — rebuilt in `member_1/model_training.ipynb`

The 25 required sections, mapped to implementation:

| § | Section | Implementation |
|---|---------|----------------|
| 1 | Objective | prose (mine) |
| 2 | Imports & Configuration | `common/` imports + seeds + config block |
| 3 | Dataset Inspection | `get_data` + `count_images` + EDA grid (real stats printed) |
| 4 | Preprocessing | backbone-preprocess-inside-graph explanation (already built) |
| 5 | Train/Val/Test Setup | `build_datasets` + fingerprints + leakage checks |
| 6 | Evaluation Metrics | `common/eval.py` wrapper + convention statement |
| 7 | TensorBoard Configuration | `ExperimentLogger` + `%tensorboard` launch |
| 8 | MobileNet Architecture | §3 diagram + trainable-params proof |
| 9–15 | Experiments 01–07 | one cell-block per experiment: md hypothesis → run → logged metrics |
| 16 | Additional experiments | only if justified (§5 rule) |
| 17 | Experiment Comparison Table | `scan_runs()` → pandas table |
| 18 | Final Model Selection | multi-metric rationale cell |
| 19 | Final Test Evaluation | single test evaluation |
| 20–22 | Curves / Confusion / ROC | 4 figures |
| 23 | Error Analysis | mining + plots + write-up prompts |
| 24 | Grad-CAM | §8 artifacts |
| 25 | Final Findings | `[FILL]`-marked summary |

## 10. Experiment comparison table (spec §19)

Columns: Experiment · Configuration · LR · Optimizer · Batch Size ·
Augmentation · Dropout/L2 · Frozen/Unfrozen Strategy · Accuracy · Precision ·
Recall · F1 · ROC-AUC · Sensitivity · Specificity · Outcome/Observation.
Filled **only** from logged runs; a missing metric is marked N/A-with-reason,
never guessed.

## 11. Report contribution draft (spec §20–§22)

`report/member1_mobilenet_contribution.md` — draft/outline only:
Methodology (architecture & why, transfer strategy, preprocessing actually
used, fine-tuning strategy) · Experiments (table + what changed/why/outcome) ·
Final Evaluation (metrics, figures, `[FILL]` interpretation) · Error Analysis ·
Explainability (+ limitations) · Limitations (experiment-supported only).
No invented citations → `[REFERENCE NEEDED]` markers. Narrative skeleton, my
rewrite before submission.

## 12. Verification & QC (spec §16–§17, §25)

- **Local (CPU, synthetic):** existing 45-test suite stays green;
  glue smoke = run full notebook flow on `make_synthetic_cell_dataset`
  (3 epochs) before any Colab session.
- **Colab runbook:** GPU runtime → Drive mount → run §1–§8 → EXP01 → verify
  TB artifacts → EXP02…07 sequentially (results decide next variable) →
  selection → final test → artifacts → zip logs to Drive.
- **Final audit:** spec §25 checklist reproduced as a table in the notebook's
  last section, each item answered with the artifact that proves it.
- Reproducibility: seed 42 everywhere, config dict per experiment, versions
  printed.

## 13. Work sequence (spec §26 — with verification gates)

1. ✅ Inspect repo (this plan §1)
2. ✅ Report what exists / what's missing / decisions (§2 → ask user)
3. Build notebook §1–§8 + EXP01 → **gate: local synthetic smoke green**
4. Colab EXP01 → **gate: TB run dir + full metric set real**
5. EXP02–07 one at a time → **gate: logbook paragraph + comparison row each**
6. Selection → single final test eval → 4 figures
7. Error analysis + Grad-CAM (3 example types)
8. Report contribution draft + comparison table + audit checklist
9. Zip TB logs to Drive; final §25 audit

## 14. Risks

| Risk | Mitigation |
|---|---|
| Fine-tuning wrecks pretrained features (catastrophic forgetting) | low-LR exp 06 first, unfreeze depth only after stability shown |
| Test-set leakage via repeated evaluation | test evaluated ONCE (§4); validation guides everything |
| Sigmoid saturation → blank Grad-CAM | logit-based implementation (already tested) |
| Colab disconnect loses logs | TB_LOGDIR on Drive + zip export |
| Label-convention bugs | Parasitized=1 fixed in `common/data.py`, verified by tests + printed EDA check |

---

**Decisions needed from you before step 3** (asked separately):
backbone version · run-name prefix · stale draft handling.

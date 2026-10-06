# Malaria CNN Classification Project Report

> Source: `Malaria-CNN-Classification-Project-` (group, 4 members).
> Member 1 contribution: MobileNetV3Small transfer learning (transfer model 1).
> This report is written in a plain, human voice. Square brackets — `[FILL]`,
> `[CHECK]`, `[REFERENCE NEEDED]` — are left only where our own words still
> need to go, not for invented numbers or invented citations.

## Abstract

This report covers a binary malaria diagnosis study on the NIH/NLM cell image
set (27,558 images, Parasitized vs Uninfected). We compare four models, and this
section summarises the one I own: a MobileNetV3Small backbone that is ImageNet
pretrained, trained first with a frozen head and then with a small amount of
fine-tuning. The group target is 90% sensitivity and 90% specificity on a
held-out test set; this model targets 95% sensitivity.

## 1. Introduction

### 1.1 Background

Malaria is still diagnosed mostly by looking at blood smears under a
microscope. That works, but it depends on the skill of the person reading the
slide, and it is slow. An automatic tool that can tell whether a cell looks
infected would help in clinics and in places where experts are short.

The dataset we use is the NIH/NLM malaria cell image set. It contains 27,558
photos of single blood cells, each labelled Parasitized or Uninfected. The two
classes are balanced, which keeps the metrics easy to read.

### 1.2 Objectives

- Build one transfer-learning model, MobileNetV3Small, and run at least 7
  meaningful experiments with TensorBoard evidence.
- Report accuracy, precision, recall, F1, ROC-AUC, sensitivity and specificity
  for every run.
- Keep the test set untouched during selection, and evaluate it exactly once,
  after the final model is chosen.
- Produce the required figures: accuracy/loss curves, confusion matrix, ROC/AUC,
  error analysis and Grad-CAM.
- Write this report in a plain, human voice, with honest `[FILL]` markers,
  no invented numbers and no invented citations.

### 1.3 Scope

This report covers the MobileNetV3Small model only. Members 2, 3 and 4 build
their own models (custom ResNet, a second transfer model, another custom CNN),
and their sections live in the same document. I do not report on their models,
and they do not report on mine. The shared split, shared evaluation code and
shared TensorBoard convention are reused from the group, read-only.

## 2. Related Work

Overview of existing approaches to malaria cell classification using machine
learning and deep learning.

[REFERENCE NEEDED — list 5+ scholarly sources, with exact page/URL and date.
Nothing here is quoted from memory.]

## 3. Methodology

### 3.1 Data Acquisition

The NIH/NLM malaria image set arrives as one zip file with two class folders:
`Parasitized/` and `Uninfected/`, 13,779 images per class. `get_data` downloads
it once and caches it in Google Drive, so later Colab sessions skip the download.

We use a single file-level split shared by every member of the group:
70% train / 15% validation / 15% test, seeded with 42. The split is by file,
not by image, so the same picture can never appear in two splits. The exact
fingerprint is printed in the notebook (Section 5) and verified by tests.

### 3.2 Data Preprocessing

- Raw 0-255 RGB input. MobileNetV3's preprocess step is a pass-through that
  already lives inside the model graph, so the data pipeline stays raw end to
  end and double-preprocessing is impossible by construction.
- Augmentation is training only: horizontal and vertical flips, 15% rotation,
  15% zoom. Validation and test are never augmented.
- Images are resized to 224x224, the minimum input Keras accepts for this
  backbone.

### 3.3 Model Architecture

```
Input (224x224x3, raw 0-255)
  -> MobileNetV3 preprocess (pass-through, inside the graph)
  -> MobileNetV3Small backbone (ImageNet pretrained)
  -> GlobalAveragePooling2D
  -> Dropout(0.3)
  -> Dense(1, sigmoid)   => P(Parasitized)
```

Why this backbone: it is the smallest ImageNet model we can use in Keras, so
each experiment is fast and the 7+ run campaign stays practical. MobileNetV3's
squeeze-and-excitation blocks suit the texture differences between the two
cell types, and picking a different family from the group's other transfer
model keeps the comparison meaningful.

### 3.4 Why transfer learning (in plain words)

Training a deep CNN from scratch would need a huge amount of data, a strong
GPU, and many hours, and it would probably overfit on our 27,558 images.
Transfer learning avoids that price. MobileNetV3Small was already trained on
ImageNet, so it already knows the general things every image is made of:
edges, colours, textures and shapes. We keep that knowledge and only train a
small head on top for our two classes — this is the frozen,
feature-extraction phase. It saves time and compute, needs far fewer
labelled images, and lowers the risk of overfitting because most of the
network never changes.

Fine-tuning comes after. We unfreeze the top of the backbone and keep
training at a learning rate 10x smaller, so the pretrained features adjust
to malaria cells slowly instead of being wiped out. Unfreezing is the key
lever: if the model keeps improving, we can unfreeze more layers; if it does
not improve, that is a sign this pretrained model does not fit our images
and a different one should be tried. Early stopping (patience 5 on
validation loss) is the other guard: it stops training before the model
starts memorising the training set.

The known risk of transfer learning is domain mismatch: ImageNet is photos
of everyday objects, our data is microscope cells, so the pretrained
features are not guaranteed to help. That is exactly what experiments 01-07
measure — frozen first, then progressively unfrozen — and the evidence, not
hope, decides.

### 3.5 Training Procedure

Two phases, one variable at a time:

- Phase 1, experiments 01-04: the backbone is frozen and only the head trains.
  This gives a clean reference for every later change.
- Phase 2, experiments 05-07: the top layers of the backbone are unfrozen at
  a learning rate 10x lower, so the pretrained features adapt without being
  destroyed.

Every experiment uses the same seed, the same batch size, and the same
balanced class weights. Early stopping watches validation loss with a patience
of 5; each run saves its best checkpoint.

### 3.6 Evaluation Metrics

- Accuracy, precision, recall, F1, ROC-AUC as usual.
- Sensitivity = recall on the Parasitized class (the clinically important
  one).
- Specificity = true-negative rate on Uninfected.
- Labels: 1 = Parasitized, 0 = Uninfected. We verify this convention on a
  hand-built case before training starts.

### 3.7 Explainability Approach

We use Grad-CAM. Because the backbone is a functional model, the last conv
layer is exposed by name and graded through the logit rather than the sigmoid,
so the gradients do not saturate on confident predictions. The heatmap shows
where the model looked, never whether its looking is medically correct, which
we say in the report.

## 4. Results

### 4.1 Data Exploration

The dataset is balanced: 13,779 parasitized and 13,779 uninfected images. The
file-level split gives about 19,290 train, 4,133 validation and 4,135 test
images. [NUMBERS NEED TO BE TAKEN FROM THE NOTEBOOK OUTPUT AND INSERTED HERE].
The sample grid shows cells that genuinely look different between the two
classes, which is why a texture model should work well.

### 4.2 Model Performance

The notebook trains all seven experiments and writes the real numbers into a
comparison table. Targets are 95% sensitivity and 90% specificity. The final
selection is multi-metric: sensitivity first, then ROC-AUC, then specificity.
Accuracy alone does not decide the winner.

[TABLE TO BE INSERTED FROM THE RUN. Columns: experiment, config, lr, dropout,
augmentation, unfrozen strategy, accuracy, precision, recall, f1, roc_auc,
sensitivity, specificity, outcome. Every row is copied from the logged run,
never typed in.]

### 4.3 Best Model Selection

[DECISION RULE + FINAL RUN NAME + THE NUMBER IT SCORES TO BE INSERTED FROM
RESULTS. Rationale in plain words: we said sensitivity matters most for a
screening tool, then the threshold-free AUC, then specificity.]

### 4.4 Detailed Evaluation

The single final test evaluation of the chosen config is in Section 19 of the
notebook. It is the only place the test set is seen. The four required figures
are produced there: accuracy/loss curves, confusion matrix, ROC/AUC, and the
combined evaluation row.

### 4.5 Explainability Results

[GRADCAM GRID + THREE EXAMPLE TYPES + A FILL INTERPRETATION. The incorrect
case, when it exists, is discussed as a case study, with the caveat that the
heatmap shows where the model looked, not why.]

## 5. Discussion

### 5.1 Interpretation of Results

What the frozen features achieved in exp 01, what each controlled change bought
or cost in exps 02-04, and what fine-tuning changed in exps 05-07. The curve
diagnosis tells us whether we over- or underfit, and which experiment answered
that symptom.

### 5.2 Comparison with Prior Work

[REFERENCE NEEDED — cite the group's or a reviewer's prior CNN report with a
real source, then a short honest comparison. No numbers copied from memory.]

### 5.3 Limitations

- Synthetic smoke evidence is not the same as the 27,558-image run; the real
  numbers are still being collected.
- Grad-CAM explains where the model looked, not whether that looks is
  medically right. A cell can look plausible in the heatmap and still be a
  wrong call.
- The model predicts cell-level appearance, not patient-level disease. A
  screening tool needs a threshold tuned for the clinic, not for this
  notebook.
- We use one backbone family. The group's other transfer model and custom
  models provide the comparison, and they are not covered here.

### 5.4 Future Work

- Tune the decision threshold per use case (screening vs confirmative).
- Add more augmentations or test-time augmentation once the run is clean.
- Enlarge the fine-tuning ablation (fewer vs more unfrozen layers) from the
  evidence so far.
- Re-run the notebook on real GPU time and save the TensorBoard logs to Drive
  for the appendix.

## 6. Conclusion

This report covers one member's model and one member's evidence. MobileNetV3
Small, a pretrained backbone trained first frozen and then with a small amount
of controlled fine-tuning, is the model we selected for its sensitivity-first
behaviour. [FINAL NUMBER ~ THE TEST RESULT TO BE INSERTED FROM THE RUN]. The
group's overall claim is limited to what this run shows on this single split,
and it should be reread after the real results land.

## References

[REFERENCE NEEDED — bibliography with exact years, page numbers and URLs for
every claim. No sources from memory.]

## Member 1 Contribution (MobileNetV3Small)

This section carries my part of the work. It is the draft of how I would
write my piece of the group report; the numbers are placeholders until my
real run lands.

### 1. What I did

- Reused the group's shared data pipeline, split and evaluation code (read
  only, exactly as the group agreed).
- Registered MobileNetV3Small as Transfer Model 1 in the group config.
- Built the transfer model in the notebook: ImageNet backbone, frozen head
  phase, then a small fine-tuning phase with a 10x lower learning rate.
- Ran 7 experiments, each changing one variable, all logged to TensorBoard
  under `logs/tensorboard/transfer_model_1/`.
- Computed accuracy, precision, recall, F1, ROC-AUC, sensitivity and
  specificity for every run on the validation set.
- Kept the test set untouched until the single final evaluation after
  selection.
- Produced the four required figures, error analysis and Grad-CAM.

### 2. My reasoning behind the choices

- I chose MobileNetV3Small because it is the smallest ImageNet backbone in
  Keras, so the 7+ experiment campaign stays fast on Colab.
- I froze the backbone for the first four runs so the head could learn on
  clean ImageNet features before I touched them.
- I only unfroze the top layers in the last three runs, and at a much
  smaller learning rate, so the pretrained weights would not be destroyed.
- I ranked runs by sensitivity first, then ROC-AUC, then specificity,
  because a screening tool must not miss infected cells.

### 3. Evidence I will bring

- The comparison table from the notebook (Section 17).
- The four final figures and the Grad-CAM grid.
- The single final test metrics.
- The TensorBoard run directories for all seven experiments.

### 4. What I still owe

- The final numbers from the real run.
- The references list (marked [REFERENCE NEEDED]).
- The plain-language interpretation of the results (marked `[FILL]`).

## Appendices

### Appendix A: Additional Figures and Tables

Copy of the comparison table, per-run TensorBoard configs, and extra error
analysis images.

### Appendix B: Code Snippets

The single experiment runner, the Grad-CAM tap, and the metrics convention
check that are shared between the notebook and the tests.

### Appendix C: Experimental Setup Details

Hardware (Colab GPU), software versions, seed 42, 224x224 input, batch size 32,
30 epochs, early stopping patience 5, and the exact data-split fingerprint.

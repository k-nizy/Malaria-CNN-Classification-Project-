"""
tests/test_eval.py — verification for common/eval.py

Strategy: train two tiny models on the synthetic dataset —
  - a GOOD model (a few epochs, should actually learn the blob signal)
  - a BAD model (zero epochs — untrained, near-chance predictions)
then verify:
  1. Metrics sanity: good model beats bad model; metrics in [0,1]
  2. Sensitivity/specificity computed correctly against a hand-built case
  3. All 4 required plots render headlessly and save to disk
  4. The single-row final-evaluation figure matches rubric layout
  5. Misclassification mining finds real errors (untrained model must err)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.data import (  # noqa: E402
    SEED,
    build_datasets,
    collect_labels,
    make_synthetic_cell_dataset,
)
from common.eval import (  # noqa: E402
    Metrics,
    evaluate_metrics,
    evaluate_model,
    final_evaluation_figure,
    find_misclassifications,
    plot_confusion_matrix,
    plot_misclassifications,
    plot_roc_curve,
    plot_training_curves,
    predictions_from_probs,
)

tf.keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# Fixtures: dataset + one good (trained) + one bad (untrained) tiny model
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = make_synthetic_cell_dataset(tmp_path_factory.mktemp("eval_data"),
                                       per_class=60, img_size=(32, 32), seed=SEED)
    train, val, test = build_datasets(root, img_size=(32, 32), batch_size=16,
                                      seed=SEED)

    # MaxPool->Flatten keeps spatial specificity: GAP would dilute the small
    # blob signal into near-constant ~0.5 probabilities (verified by debug run)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    # 25 epochs: enough to learn the synthetic blob AND calibrate the
    # threshold (shorter runs left all probs on one side of 0.5)
    history = model.fit(train, validation_data=val, epochs=25, verbose=0)

    bad = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(4, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    bad.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])
    return train, val, test, model, history, bad


# ----------------------------------------------------------------------------
# 1. Metric sanity
# ----------------------------------------------------------------------------

def test_good_model_beats_untrained(env):
    """
    Compare the trained vs untrained model the way the report should:
    NOT by raw accuracy (a degenerate all-negative classifier scores 72%
    on a skewed test set!) but by ranking quality (AUC) and balanced
    performance (F1, balanced accuracy) — plus proof that SOME decision
    threshold separates the classes well.
    """
    from sklearn.metrics import roc_auc_score

    train, val, test, model, history, bad = env
    m_good, y_true, y_prob = evaluate_model(model, test)
    m_bad, _, _ = evaluate_model(bad, test)

    # 1. ranking quality: trained model must separate classes
    assert m_good.roc_auc >= 0.9, f"auc={m_good.roc_auc}"

    # 2. degenerate baseline has F1 = 0; trained model must do real work
    assert m_good.f1 > m_bad.f1, f"good={m_good.format()} vs bad={m_bad.format()}"

    # 3. balanced accuracy (sens+spec)/2 must beat chance for the good model
    bal_acc = (m_good.sensitivity + m_good.specificity) / 2
    assert bal_acc > 0.6, f"balanced acc={bal_acc}"

    # 4. a well-chosen threshold (as the ROC curve shows) gets high accuracy
    best_acc = max(
        evaluate_metrics(y_true, y_prob, threshold=t).accuracy
        for t in np.linspace(0.05, 0.95, 19)
    )
    assert best_acc >= 0.85, (
        f"AUC={m_good.roc_auc:.3f} implies a good threshold exists; best_acc={best_acc:.3f}"
    )


def test_metrics_in_unit_range(env):
    train, val, test, model, history, bad = env
    m, _, _ = evaluate_model(model, test)
    for v in (m.accuracy, m.precision, m.recall, m.f1, m.roc_auc, m.specificity):
        assert 0.0 <= v <= 1.0
    assert m.sensitivity == m.recall            # clinical alias holds


def test_hand_built_metrics_case():
    """
    Hand-verified example: 4 parasitized (true=1), 4 uninfected (true=0).
    Predictions: 3 TP, 1 FN -> sensitivity = 3/4 = 0.75
                 2 FP, 3 TN -> specificity = 3/5... (TN=3, FP=2 over 5 uninf)
    precision = 3/5, recall = 3/4, f1 = 2*0.6*0.75/(0.6+0.75)
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.55, 0.2, 0.1, 0.05])
    m = evaluate_metrics(y_true, y_prob)
    assert m.recall == pytest.approx(0.75)
    assert m.sensitivity == m.recall
    assert m.specificity == pytest.approx(0.6)          # TN=3, FP=2
    assert m.precision == pytest.approx(0.6)            # TP=3, FP=2
    expected_f1 = 2 * 0.6 * 0.75 / (0.6 + 0.75)
    assert m.f1 == pytest.approx(expected_f1)
    assert 0.5 < m.roc_auc <= 1.0


def test_threshold_effect(env):
    """A low threshold should raise sensitivity (recall) — clinical trade-off."""
    train, val, test, model, history, bad = env
    m_lo, _, _ = evaluate_model(model, test, threshold=0.1)
    m_hi, _, _ = evaluate_model(model, test, threshold=0.9)
    assert m_lo.recall >= m_hi.recall


# ----------------------------------------------------------------------------
# 2. The 4 required plots render & save
# ----------------------------------------------------------------------------

def test_training_curves_plot(env, tmp_path):
    train, val, test, model, history, bad = env
    fig = plot_training_curves(history)
    out = tmp_path / "curves.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


def test_confusion_matrix_plot(env, tmp_path):
    train, val, test, model, history, bad = env
    m, y_true, y_prob = evaluate_model(model, test)
    fig = plot_confusion_matrix(y_true, predictions_from_probs(y_prob))
    out = tmp_path / "cm.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


def test_roc_plot(env, tmp_path):
    train, val, test, model, history, bad = env
    m, y_true, y_prob = evaluate_model(model, test)
    fig = plot_roc_curve(y_true, y_prob)
    out = tmp_path / "roc.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


def test_final_evaluation_figure(env, tmp_path):
    """The rubric's single-row 4-plot layout for a model's final version."""
    train, val, test, model, history, bad = env
    fig, m, y_true, y_prob = final_evaluation_figure(model, history, test,
                                                     save_path=str(tmp_path / "final.png"))
    out = tmp_path / "final.png"
    assert out.exists() and out.stat().st_size > 20_000
    assert 0 <= m.accuracy <= 1 and m.sensitivity == m.recall
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Error-analysis mining
# ----------------------------------------------------------------------------

def test_find_misclassifications_finds_real_errors(env):
    """The untrained model must misclassify; mining must surface its errors."""
    train, val, test, model, history, bad = env
    imgs, trues, preds, probs = find_misclassifications(bad, test, n=3)
    assert len(imgs) == len(trues) == len(preds) == len(probs)
    assert len(imgs) > 0, "untrained model should have errors on test set"
    for yt, yp in zip(trues, preds):
        assert yt != yp                       # every returned case IS an error


def test_misclassification_plot(env, tmp_path):
    train, val, test, model, history, bad = env
    imgs, trues, preds, probs = find_misclassifications(bad, test, n=3)
    fig = plot_misclassifications(imgs, trues, preds, probs)
    out = tmp_path / "errors.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 5_000
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3b. Display labels must match the clinical label convention (post-flip)
# ----------------------------------------------------------------------------

def test_display_labels_match_convention():
    """After the Parasitized=1 flip, CLASS_NAMES[value] shows SWAPPED names.
    Figures must index the label-value-ordered list instead."""
    from common.eval import LABEL_NAMES_BY_VALUE
    assert LABEL_NAMES_BY_VALUE == ["Uninfected", "Parasitized"]

    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.8])
    m = evaluate_metrics(y_true, y_prob)
    row = m.as_row()
    # clinical naming lands in the table row (report §5.1 requirement)
    assert "sensitivity" in row and "specificity" in row
    assert row["sensitivity"] == 1.0 and row["specificity"] == 1.0

    # and the confusion-matrix figure carries the correct axis names
    fig = plot_confusion_matrix(np.array([0, 1, 1, 0]),
                                np.array([0, 1, 0, 1]))
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["Pred Uninfected", "Pred Parasitized"], (
        f"confusion-matrix columns swapped: {labels}"
    )
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Metrics dataclass table-row formatting
# ----------------------------------------------------------------------------

def test_metrics_as_row():
    m = Metrics(accuracy=0.95, precision=0.94, recall=0.97, f1=0.955,
                roc_auc=0.98, specificity=0.93)
    row = m.as_row()
    assert row["recall"] == row["sensitivity"] == 0.97
    assert set(row) >= {"accuracy", "precision", "recall", "f1", "roc_auc",
                        "specificity", "sensitivity"}

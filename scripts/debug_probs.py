"""Debug: inspect the probability distributions the tiny model produces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
from common.data import SEED, build_datasets, make_synthetic_cell_dataset
from common.eval import collect_predictions

tf.keras.utils.set_random_seed(SEED)

root = make_synthetic_cell_dataset("data/_debug", per_class=60, img_size=(32, 32), seed=SEED)
train, val, test = build_datasets(root, img_size=(32, 32), batch_size=16, seed=SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=12, verbose=0)

y_true, y_prob = collect_predictions(model, test)
print("\ny_true class counts:", np.bincount(y_true.astype(int)))
print("y_prob for true=1 (parasitized): min=%.4f max=%.4f mean=%.4f" % (
    y_prob[y_true == 1].min(), y_prob[y_true == 1].max(), y_prob[y_true == 1].mean()))
print("y_prob for true=0 (uninfected): min=%.4f max=%.4f mean=%.4f" % (
    y_prob[y_true == 0].min(), y_prob[y_true == 0].max(), y_prob[y_true == 0].mean()))
print("all probs:", np.round(np.sort(y_prob), 3))

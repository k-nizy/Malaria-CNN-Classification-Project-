"""Debug the all-zero heatmap: inspect conv activations + gradient stats."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
from common.data import SEED, build_datasets, make_synthetic_cell_dataset
from common.models import build_custom_resnet

tf.keras.utils.set_random_seed(SEED)

root = make_synthetic_cell_dataset("data/_debug", per_class=60, img_size=(32, 32), seed=SEED)
train, val, test = build_datasets(root, img_size=(32, 32), batch_size=16, seed=SEED)

model = build_custom_resnet(img_size=(32, 32), dropout=0.2, width_mult=0.25)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(train, validation_data=val, epochs=30, verbose=0)
print("final val_acc:", h.history["val_accuracy"][-1])

# one blob image
rng = np.random.default_rng(7)
img = rng.integers(150, 190, size=(32, 32, 3)).astype("uint8")
yy, xx = np.mgrid[:32, :32]
img[(yy - 14) ** 2 + (xx - 18) ** 2 <= 25] = (30, 30, 60)
x = tf.convert_to_tensor(img[None].astype("float32"))

model._last_conv_features = None
prob = model(x, training=False)
conv = model._last_conv_features
print("prob:", float(tf.squeeze(prob).numpy()))
print("conv stats: min=%.4f max=%.4f mean=%.4f" % (
    float(tf.reduce_min(conv)), float(tf.reduce_max(conv)),
    float(tf.reduce_mean(conv))))

with tf.GradientTape() as tape:
    tape.watch(conv)
    y = model.gap(conv)
    y = model.dropout(y, training=False)
    z = tf.matmul(y, model.head.kernel) + model.head.bias   # LOGIT
    score = tf.squeeze(z, axis=-1)
grads = tape.gradient(score, conv)
print("grad stats: none?", grads is None,
      "" if grads is None else "min=%.6f max=%.6f mean=%.6f" % (
          float(tf.reduce_min(grads)), float(tf.reduce_max(grads)),
          float(tf.reduce_mean(grads))))
if grads is not None:
    w = tf.reduce_mean(grads, axis=(0, 1))
    print("alpha_k: min=%.6f max=%.6f  #positive=%d/%d" % (
        float(tf.reduce_min(w)), float(tf.reduce_max(w)),
        int(tf.reduce_sum(tf.cast(w > 0, tf.int32))), w.shape[0]))
    cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1))
    print("cam stats: min=%.4f max=%.4f" % (
        float(tf.reduce_min(cam)), float(tf.reduce_max(cam))))
    print("head kernel stats: min=%.4f max=%.4f" % (
        float(tf.reduce_min(model.head.kernel)),
        float(tf.reduce_max(model.head.kernel))))

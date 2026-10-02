"""Minimal repro for the ResidualBlock call failure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
from common.models import build_custom_resnet, ResidualBlock

# 1. bare block call
blk = ResidualBlock(16)
x = np.random.rand(2, 16, 16, 16).astype("float32")
try:
    y = blk(x, training=False)
    print("stride1 block OK:", y.shape)
except Exception as e:
    print("stride1 block FAILED:")
    import traceback; traceback.print_exc()

blk2 = ResidualBlock(16, stride=2)
try:
    y = blk2(x, training=False)
    print("stride2 block OK:", y.shape)
except Exception as e:
    print("stride2 block FAILED:")
    import traceback; traceback.print_exc()

# 2. full model call
m = build_custom_resnet(img_size=(32, 32))
try:
    out = m(np.random.rand(1, 32, 32, 3).astype("float32"), training=False)
    print("model OK:", out.shape)
    print("tap:", m._last_conv_features.shape)
except Exception:
    print("model FAILED:")
    import traceback; traceback.print_exc()

# 3. param counts for the levers test
small = build_custom_resnet(img_size=(32, 32), width_mult=0.5)
big = build_custom_resnet(img_size=(32, 32), width_mult=1.0)
print("params small(0.5):", small.count_params(), " big(1.0):", big.count_params())

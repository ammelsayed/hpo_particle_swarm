import os
import xgboost as xgb
print("xgboost", xgb.__version__)
if hasattr(xgb, "build_info"):
    info = xgb.build_info()
    for k in sorted(info):
        print(f"  {k}: {info[k]}")
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH"))
import numpy as np
X = np.random.rand(200, 4).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.float32)
d = xgb.DMatrix(X, label=y)
for device in ["cuda", "cuda:0", "cpu"]:
    try:
        b = xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_depth": 2, "verbosity": 1},
            d,
            num_boost_round=2,
        )
        print(f"device={device} OK rounds={b.num_boosted_rounds()}")
    except Exception as e:
        print(f"device={device} FAIL: {e}")

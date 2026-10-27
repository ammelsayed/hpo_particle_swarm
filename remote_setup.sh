#!/bin/bash
set -euo pipefail
VENV=/data/ammelsayed/venv
DIR=/data/ammelsayed/particle_swarm
"$VENV/bin/pip" install 'numpy>=2.1' -q
"$VENV/bin/python" -c "import numpy, mealpy, xgboost, sklearn; print('numpy', numpy.__version__); print('mealpy', mealpy.__version__); print('xgboost', xgboost.__version__); print('sklearn', sklearn.__version__)"
# GPU smoke test
"$VENV/bin/python" - <<'PY'
import numpy as np, xgboost as xgb
X = np.random.rand(500, 8).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.float32)
d = xgb.DMatrix(X, label=y)
b = xgb.train({"objective":"binary:logistic","tree_method":"hist","device":"cuda","max_depth":2,"verbosity":0}, d, num_boost_round=5)
print("GPU OK", b.num_boosted_rounds())
PY
echo "Setup complete."

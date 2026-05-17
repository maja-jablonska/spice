#!/usr/bin/env bash
# Create SUPPNet conda env and install the package in editable mode.
# Run from anywhere; clones/uses ~/code/suppnet by default.
set -euo pipefail

SUPPNET_REPO="${SUPPNET_REPO:-$HOME/code/suppnet}"
ENV_NAME="${SUPPNET_ENV_NAME:-suppnet-env}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "$SUPPNET_REPO/.git" ]]; then
  echo "Cloning SUPPNet into $SUPPNET_REPO"
  git clone https://github.com/RozanskiT/suppnet.git "$SUPPNET_REPO"
fi

ENV_FILE="$SUPPNET_REPO/environment-macos-arm64.yml"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  :
elif [[ -f "$SUPPNET_REPO/environment.yml" ]]; then
  ENV_FILE="$SUPPNET_REPO/environment.yml"
  echo "Using upstream environment.yml (non-arm64 Mac or Linux)"
else
  ENV_FILE="$SUPPNET_REPO/environment-macos-arm64.yml"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists — activating and reinstalling suppnet"
else
  echo "Creating conda env '$ENV_NAME' from $ENV_FILE"
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

conda activate "$ENV_NAME"
pip install -e "$SUPPNET_REPO"
# extras for harps_suppnet_normalize.ipynb
pip install -q ipykernel zarr tqdm

# TF 2.4 (upstream environment.yml) aborts on Apple Silicon during model.predict.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Installing tensorflow-macos 2.9.x for Apple Silicon…"
  pip install -q 'tensorflow-macos>=2.9.2,<2.10' 'tensorflow-metal>=0.5.1,<0.6'
fi

python - <<'PY'
import numpy as np
from suppnet.NN_utility import get_suppnet
import tensorflow as tf

ver = tuple(int(x) for x in tf.__version__.split('.')[:2])
if ver < (2, 9):
    raise SystemExit(
        f'TensorFlow {tf.__version__} is too old for Apple Silicon SUPPNet predict'
    )
print('OK: tensorflow', tf.__version__)
print('OK: suppnet import; loading model weights…')
nn = get_suppnet(resampling_step=0.05, norm_only=False, which_weights='active')
print('OK: SUPPNet model loaded')
# Smoke-test predict (TF 2.4 dies here on arm64).
X = np.random.rand(2, 8192, 1).astype(np.float32)
_ = nn.model.predict(X)
print('OK: SUPPNet predict smoke test passed')
PY

echo ""
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
echo "  export SUPPNET_REPO=$SUPPNET_REPO"
echo ""
echo "Register Jupyter kernel (optional):"
echo "  python -m ipykernel install --user --name $ENV_NAME --display-name 'Python (suppnet)'"
echo ""
echo "Note: harps_read.ipynb writes Zarr v3; suppnet-env (Py 3.8) uses zarr 2.x."
echo "harps_suppnet_normalize.ipynb auto-converts via 'conda run -n astro' if that env exists."

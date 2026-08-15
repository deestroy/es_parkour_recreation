#!/usr/bin/env bash
# Replace the CUDA torch (pulled in by torchvision resolution) with ROCm builds.
set -euo pipefail
PIP=~/esparkour_venv/bin/pip
PY=~/esparkour_venv/bin/python

CUDA_PKGS=$($PIP list 2>/dev/null | awk '{print $1}' | grep -E '^(nvidia-|cuda-)' | tr '\n' ' ' || true)
$PIP uninstall -y torch torchvision triton $CUDA_PKGS || true
$PIP install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
$PY - <<'EOF'
import torch, torchvision, spikingjelly
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("gpu_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul_ok", float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo FIX_DONE

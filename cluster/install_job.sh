#!/bin/bash
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"
#SBATCH --partition=gpu-bermano
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=install_job-%j.out

VENV=${PROJECT_DIR}/.venv
CACHE=${PROJECT_DIR}/.wheel_cache

rm -rf /tmp/torch_install_local /tmp/tv_install_local
mkdir -p /tmp/torch_install_local /tmp/tv_install_local

"$VENV/bin/python" -m pip install --no-deps --target /tmp/torch_install_local \
  "$CACHE/torch-2.3.1+cu121-cp312-cp312-linux_x86_64.whl"
echo "TORCH_LOCAL_INSTALL_DONE"

"$VENV/bin/python" -m pip install --no-deps --target /tmp/tv_install_local \
  "$CACHE/torchvision-0.18.1+cu121-cp312-cp312-linux_x86_64.whl"
echo "TORCHVISION_LOCAL_INSTALL_DONE"

tar -cf /tmp/torch_local.tar -C /tmp/torch_install_local .
tar -cf /tmp/tv_local.tar -C /tmp/tv_install_local .
echo "TAR_DONE"

cp /tmp/torch_local.tar /tmp/tv_local.tar "$CACHE/"
echo "COPY_DONE"

# Quick sanity check for the compute node's own GPU before we trust it.
LD_LIBRARY_PATH="/tmp/torch_install_local/torch/lib:$LD_LIBRARY_PATH" \
PYTHONPATH="/tmp/torch_install_local:/tmp/tv_install_local" \
"$VENV/bin/python" -c "
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul ok, sum:', float(y.sum()))
"
echo "JOB_DONE"

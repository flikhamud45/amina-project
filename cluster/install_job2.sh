#!/bin/bash
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"
#SBATCH --partition=gpu-bermano
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=install_job2-%j.out

VENV=${PROJECT_DIR}/.venv

"$VENV/bin/pip" install --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 torch torchvision
echo "PIP_INSTALL_DONE"

"$VENV/bin/python" -c "
import torch, torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul ok, sum:', float(y.sum()))
"
echo "JOB_DONE"

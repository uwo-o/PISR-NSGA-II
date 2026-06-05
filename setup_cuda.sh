#!/bin/bash
# setup_cuda_env.sh - Configura un entorno uv con Python 3.12 y PyTorch con soporte sm_61

set -e

echo "=== 1. Creando entorno virtual aislado con Python 3.12 ==="
uv venv --python 3.12

echo "=== 2. Activando entorno ==="
source .venv/bin/activate

echo "=== 3. Instalando PyTorch Oficial con soporte CUDA 12.1 (incluye sm_61) ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. Instalando DeepXDE y dependencias del pipeline ==="
uv pip install deepxde matplotlib pandas numpy

echo "=== 5. Verificando compatibilidad de la GPU ==="
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    # Intentamos alojar un tensor en la GPU
    try:
        x = torch.randn(2, 2, device='cuda')
        print('[SUCCESS] GPU is fully functional and ready to train PINNs!')
    except Exception as e:
        print('[ERROR] Failed to use GPU:', e)
"

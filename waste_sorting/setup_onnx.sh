#!/bin/bash
# Install ONNX Runtime GPU 1.10.0 for Jetson Nano (JetPack 4.6 / Python 3.6)

echo "============================================="
echo "  INSTALLING ONNX RUNTIME GPU (CUDA)"
echo "============================================="

# 1. Download Wheel from NVIDIA
echo "[1/2] Downloading onnxruntime-gpu 1.10.0..."
wget https://nvidia.box.com/shared/static/pms2soutn4u3cx5cqm9qibdad9qdgkdq.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

# 2. Install
echo "[2/2] Installing..."
python3 -m pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
rm onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

echo "DONE! Run 'python3 inference.py' to use CUDA."

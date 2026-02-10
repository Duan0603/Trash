#!/bin/bash
# =============================================
# Jetson Nano Setup Script for Waste Sorting
# =============================================
# Run this script on Jetson Nano to install dependencies
# Usage: chmod +x setup_jetson.sh && ./setup_jetson.sh

echo "========================================="
echo "  Waste Sorting - Jetson Nano Setup"
echo "========================================="

# 1. Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -y

# 2. Install Python dependencies
echo "[2/5] Installing Python packages..."
pip3 install --upgrade pip
pip3 install ultralytics opencv-python-headless numpy

# 3. Install Jetson GPIO library
echo "[3/5] Installing Jetson GPIO..."
pip3 install Jetson.GPIO
sudo groupadd -f -r gpio
sudo usermod -aG gpio $USER

# 4. Install GStreamer (for CSI camera)
echo "[4/5] Installing GStreamer..."
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    libgstreamer1.0-dev

# 5. Set permissions
echo "[5/5] Setting permissions..."
sudo chmod 666 /dev/video0 2>/dev/null || true

echo ""
echo "========================================="
echo "  Setup hoàn tất!"
echo "========================================="
echo ""
echo "Bước tiếp theo:"
echo "  1. Copy thư mục 'waste_sorting' sang Jetson Nano"
echo "  2. cd waste_sorting/scripts"
echo "  3. python3 inference.py"
echo ""
echo "Lần đầu chạy sẽ tự động export TensorRT (~5-10 phút)"
echo "Các lần sau sẽ chạy nhanh hơn nhiều."

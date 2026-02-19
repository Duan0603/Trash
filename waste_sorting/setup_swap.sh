#!/bin/bash
# Script to add 4GB Swap file to Jetson Nano
# Fixes freezing/lagging issues due to low RAM

echo "============================================="
echo "  CREATING 4GB SWAP FILE"
echo "============================================="

# check if swap exists
grep -q "swapfile" /etc/fstab

# if not then create it
if [ $? -ne 0 ]; then
  echo "[INFO] Creating swapfile..."
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
  echo "[SUCCESS] Swap created!"
else
  echo "[INFO] Swapfile already exists."
fi

# check swap
free -h

echo ""
echo "============================================="
echo "  PERFORMANCE MODE"
echo "============================================="
echo "Running jetson_clocks to max out performance..."
sudo jetson_clocks

echo "[INFO] Setup complete. System should be more stable."

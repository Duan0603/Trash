from ultralytics import YOLO

# Load model
model = YOLO('../models/waste_sorter/weights/best.pt')

# Export to ONNX with specific settings for OpenCV DNN / Jetson
model.export(
    format='onnx',
    imgsz=480,          # Fixed size matching inference.py
    opset=12,           # Lower opset for better compatibility
    dynamic=False,      # Force static shape (CRITICAL for OpenCV DNN)
    simplify=True       # Simplify graph
)

print("Export complete! Copy 'best.onnx' to Jetson Nano.")

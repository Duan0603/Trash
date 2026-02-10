# Smart Waste Sorting System (YOLOv8)

This project implements an object detection system for sorting waste (Plastic, Metal, Other) on a conveyor belt using a Jetson Nano.

## Project Structure
```
waste_sorting/
├── datasets/          # Place your images and labels here
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
├── models/            # Trained models and exports
├── scripts/
│   ├── train.py       # Training script
│   └── inference.py   # Real-time inference script
└── data.yaml          # YOLO configuration
```

## 1. Setup Dataset
1. Collect images of waste on your conveyor belt.
2. Label them using tools like [LabelImg](https://github.com/heartexlabs/labelImg) or Roboflow.
3. Organize them into `datasets/images/train`, `datasets/images/val`, and corresponding label folders.
4. Update `datasets` paths in `data.yaml` if necessary (current config assumes relative path `../datasets`).

## 2. Train the Model
Run the training script (requires a GPU machine for speed, can be done on PC before deploying to Jetson):
```bash
cd waste_sorting/scripts
python train.py
```
This will:
- Download `yolov8n.pt` base model.
- Train for 50 epochs.
- Save the best model to `waste_sorting/models/waste_sorter/weights/best.pt`.
- Export it to ONNX format.

## 3. Deployment on Jetson Nano
1. Copy the `waste_sorting` folder to your Jetson Nano.
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```
3. Run inference:
   ```bash
   cd waste_sorting/scripts
   python inference.py
   ```

## 4. Hardware Integration
The `inference.py` script contains a `trigger_actuator(object_class)` function.
- Edit this function to implement your specific GPIO control logic (e.g., using `Jetson.GPIO`).
- Adjust `ROI_X`, `ROI_Y`, etc., in `inference.py` to match your camera's view of the conveyor belt.

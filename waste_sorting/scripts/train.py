from ultralytics import YOLO

def train_model():
    # Start FRESH training from pretrained model
    model = YOLO('yolov8n.pt')

    # Train with 3 classes
    results = model.train(
        data='../data.yaml',
        epochs=50,  # 50 epochs for ~450 images
        imgsz=640,
        device='cpu',
        project='../models',
        name='waste_sorter',
        exist_ok=True,
        batch=8,
        fliplr=0.5,
        flipud=0.5,
        degrees=15,
        mosaic=0.5,
        patience=20,
    )

    # Export the model
    model.export(format='onnx')

if __name__ == '__main__':
    train_model()

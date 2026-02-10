"""
Waste Sorting Inference - Hybrid Mode
1. Checks for CUDA/PyTorch (Best performance on Jetson if installed)
2. Fallback to OpenCV DNN (Good CPU performance)
"""
import time
import os
import sys
import collections
import platform
import numpy as np

# ================== CONFIG ==================
# Classes
CLASS_NAMES = ["plastic", "metal", "other"]
CLASS_COLORS = [(0, 255, 0), (0, 165, 255), (128, 128, 128)]

# ROI
ROI_X1, ROI_Y1 = 150, 100
ROI_X2, ROI_Y2 = 490, 380

# Motion
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 3000
BG_LEARN_FRAMES = 30
IMGSZ = 480
CONF_THRESHOLD = 0.25

# GPIO
SERVO_PLASTIC = 33
SERVO_METAL = 32
SERVO_OTHER = 35

IS_JETSON = os.path.exists('/etc/nv_tegra_release')
HAS_DISPLAY = True
if 'DISPLAY' not in os.environ: HAS_DISPLAY = False

# ================== BACKENDS ==================
def try_load_pytorch():
    """Try loading Ultralytics YOLO (PyTorch backend)"""
    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] PyTorch CUDA detected! Using GPU.")
            from ultralytics import YOLO
            model = YOLO('../models/waste_sorter/weights/best.pt')
            model.to('cuda')
            return model, 'pytorch'
    except ImportError:
        pass
    return None, None

def try_load_opencv():
    """Fallback: Load OpenCV DNN (ONNX backend)"""
    import cv2
    onnx_path = 'best.onnx' if os.path.exists('best.onnx') else '../models/waste_sorter/weights/best.onnx'
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        return None, None

    print(f"[INFO] Using OpenCV DNN (CPU optimized). Model: {onnx_path}")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, 'opencv'

# ================== UTILS ==================
def setup_gpio():
    if not IS_JETSON: return None
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        for p in [SERVO_PLASTIC, SERVO_METAL, SERVO_OTHER]:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)
        return GPIO
    except: return None

def trigger(cls, gpio):
    if not gpio: return
    pins = {"plastic": SERVO_PLASTIC, "metal": SERVO_METAL, "other": SERVO_OTHER}
    if cls in pins:
        try:
            gpio.output(pins[cls], gpio.HIGH)
            time.sleep(0.3)
            gpio.output(pins[cls], gpio.LOW)
        except: pass

def open_camera():
    import cv2
    if IS_JETSON:
        gst = f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGR ! appsink drop=1"
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened(): return cap
        return cv2.VideoCapture(0, cv2.CAP_V4L2)
    return cv2.VideoCapture(0)

# ================== MAIN ==================
def main():
    import cv2
    print("="*40)
    
    # 1. Try PyTorch (CUDA)
    model, backend = try_load_pytorch()
    
    # 2. If no CUDA, try OpenCV (CPU)
    if not model:
        model, backend = try_load_opencv()
    
    if not model:
        print("[ERROR] No valid model loaded!")
        return

    print(f"[INFO] Backend active: {backend.upper()}")
    
    gpio = setup_gpio()
    cap = open_camera()
    if not cap.isOpened(): return

    # Background learn
    print("[INFO] Learning background...")
    bg_acc = None
    for _ in range(BG_LEARN_FRAMES):
        ret, frame = cap.read()
        if ret:
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if bg_acc is None: bg_acc = gray.astype(float)
            else: cv2.accumulateWeighted(gray, bg_acc, 0.5)
    
    bg_gray = cv2.convertScaleAbs(bg_acc)
    print("[INFO] Ready!")
    
    history = collections.deque(maxlen=5)
    last_act = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t0 = time.time()
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        
        # Motion
        diff = cv2.absdiff(gray, bg_gray)
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(thresh) > MOTION_MIN_AREA:
            # Run Inference
            results = []
            if backend == 'pytorch':
                # PyTorch Inference
                res = model.predict(roi, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    results.append((x1,y1,x2,y2,conf,cls))
            else:
                # OpenCV Inference
                blob = cv2.dnn.blobFromImage(roi, 1/255.0, (IMGSZ, IMGSZ), swapRB=True, crop=False)
                model.setInput(blob)
                outs = model.forward()
                # Simplified postprocess for readability (assume single output)
                outs = outs[0]
                if outs.shape[0] < outs.shape[1]: outs = outs.T
                h, w = roi.shape[:2]
                rx, ry = w/IMGSZ, h/IMGSZ
                for det in outs:
                    scores = det[4:]
                    cls = np.argmax(scores)
                    conf = scores[cls]
                    if conf > CONF_THRESHOLD:
                        cx, cy, bw, bh = det[:4]
                        x1 = int((cx - bw/2) * rx)
                        y1 = int((cy - bh/2) * ry)
                        x2 = int((cx + bw/2) * rx)
                        y2 = int((cy + bh/2) * ry)
                        results.append((x1,y1,x2,y2,conf,cls))

            # Draw & Logic
            best_det = None
            max_conf = 0
            
            for (x1,y1,x2,y2,conf,cls) in results:
                gx1, gy1 = x1+ROI_X1, y1+ROI_Y1
                gx2, gy2 = x2+ROI_X1, y2+ROI_Y1
                color = CLASS_COLORS[cls]
                cv2.rectangle(frame, (gx1,gy1), (gx2,gy2), color, 2)
                if conf > max_conf:
                    max_conf = conf
                    best_det = CLASS_NAMES[cls]
            
            history.append(best_det)
        else:
            history.append(None)
            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0,255,255), 2)

        # Actuator
        if len(history) == 5:
            c = collections.Counter([h for h in history if h])
            if c:
                top = c.most_common(1)[0]
                if top[1] >= 2 and (time.time() - last_act) > 1.0:
                    trigger(top[0], gpio)
                    last_act = time.time()
                    cv2.putText(frame, f"SORT: {top[0].upper()}", (20, 400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if HAS_DISPLAY:
            cv2.imshow("Waste Sorting", frame)
            if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    if HAS_DISPLAY: cv2.destroyAllWindows()
    if gpio: 
        try: gpio.cleanup()
        except: pass

if __name__ == "__main__":
    main()
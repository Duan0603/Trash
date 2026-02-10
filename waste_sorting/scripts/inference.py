import cv2
import time
import os
import sys
import collections
import platform

# ================== DETECT PLATFORM ==================
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or 'aarch64' in platform.machine()

# ================== CONFIG ==================
# Model path - try TensorRT first, then ONNX, then PT
if IS_JETSON:
    MODEL_PRIORITY = [
        '../models/waste_sorter/weights/best.engine',   # TensorRT (fastest on Jetson)
        '../models/waste_sorter/weights/best.onnx',      # ONNX (medium)
        '../models/waste_sorter/weights/best.pt',        # PyTorch (slowest)
    ]
    IMGSZ = 320       # Smaller image for Jetson performance
    DEVICE = 0         # GPU 0 on Jetson
else:
    MODEL_PRIORITY = [
        '../models/waste_sorter/weights/best.pt',
    ]
    IMGSZ = 640
    DEVICE = 'cpu'

CONF_THRESHOLD = 0.35
WIDTH, HEIGHT = 640, 480

# Classes: 0=plastic, 1=metal, 2=other
CLASSES = [0, 1, 2]
CLASS_NAMES = ["plastic", "metal", "other"]
CLASS_COLORS = [(0, 255, 0), (0, 165, 255), (128, 128, 128)]  # Green, Orange, Gray

# Vùng ROI
ROI_X1, ROI_Y1 = 150, 100
ROI_X2, ROI_Y2 = 490, 380

# Smoothing
HISTORY_LEN = 5
detection_history = collections.deque(maxlen=HISTORY_LEN)
last_trigger_time = 0
TRIGGER_COOLDOWN = 0.5

# GPIO pins for Jetson Nano actuator control
SERVO_PIN_PLASTIC = 33   # PWM pin for plastic sorting
SERVO_PIN_METAL = 32     # PWM pin for metal sorting
SERVO_PIN_OTHER = 35     # PWM pin for other sorting
# ============================================

def find_model():
    """Find the best available model file."""
    for path in MODEL_PRIORITY:
        if os.path.exists(path):
            return path
    print("[ERROR] Không tìm thấy model nào!")
    print("Các đường dẫn đã kiểm tra:")
    for p in MODEL_PRIORITY:
        print(f"  - {os.path.abspath(p)}")
    return None

def setup_gpio():
    """Setup GPIO pins on Jetson Nano for actuator control."""
    if not IS_JETSON:
        return None
    
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        # Setup output pins
        for pin in [SERVO_PIN_PLASTIC, SERVO_PIN_METAL, SERVO_PIN_OTHER]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        print("[INFO] GPIO đã khởi tạo thành công.")
        return GPIO
    except ImportError:
        print("[WARNING] Thư viện Jetson.GPIO không tìm thấy. Bỏ qua GPIO.")
        return None
    except Exception as e:
        print(f"[WARNING] Lỗi GPIO: {e}")
        return None

def trigger_actuator(class_name, confidence, gpio=None):
    """Trigger the sorting actuator based on detected class."""
    print(f">> PHAT HIEN: [{class_name.upper()}] ({confidence:.0%})")
    
    if gpio is None or not IS_JETSON:
        return
    
    # Map class to GPIO pin
    pin_map = {
        "plastic": SERVO_PIN_PLASTIC,
        "metal": SERVO_PIN_METAL,
        "other": SERVO_PIN_OTHER,
    }
    
    pin = pin_map.get(class_name)
    if pin:
        try:
            gpio.output(pin, gpio.HIGH)
            time.sleep(0.3)  # Servo activation time
            gpio.output(pin, gpio.LOW)
        except Exception as e:
            print(f"[ERROR] GPIO trigger failed: {e}")

def open_camera():
    """Open camera - handles both Jetson CSI and USB cameras."""
    if IS_JETSON:
        # Try GStreamer pipeline for CSI camera first (fastest on Jetson)
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={WIDTH}, height={HEIGHT}, format=BGR ! "
            f"appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[INFO] Camera CSI đã mở (GStreamer).")
            return cap
        
        # Fallback to USB camera on Jetson
        print("[INFO] Không tìm thấy CSI camera, thử USB camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            print("[INFO] USB camera đã mở.")
            return cap
    else:
        # Windows / Desktop
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            return cap
    
    print("[ERROR] Không thể mở camera!")
    return None

def export_tensorrt(model_path):
    """Export model to TensorRT format for Jetson Nano (one-time setup)."""
    if not IS_JETSON:
        print("[INFO] TensorRT export chỉ chạy trên Jetson Nano.")
        return model_path
    
    engine_path = model_path.replace('.pt', '.engine')
    if os.path.exists(engine_path):
        print(f"[INFO] TensorRT engine đã tồn tại: {engine_path}")
        return engine_path
    
    print("[INFO] Đang export TensorRT engine (chỉ chạy 1 lần, mất ~5-10 phút)...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format='engine', imgsz=IMGSZ, half=True, device=0)
        print(f"[INFO] Export thành công: {engine_path}")
        return engine_path
    except Exception as e:
        print(f"[WARNING] TensorRT export thất bại: {e}")
        return model_path

def get_stabilized_class(history):
    if not history: return None
    valid = [h for h in history if h is not None]
    if not valid: return None
    counts = collections.Counter(valid)
    most_common = counts.most_common(1)[0]
    if most_common[1] >= 2:
        return most_common[0]
    return None

def main():
    global last_trigger_time
    
    print("=" * 50)
    print("  WASTE SORTING SYSTEM")
    print(f"  Platform: {'JETSON NANO' if IS_JETSON else 'DESKTOP'}")
    print("=" * 50)
    
    # Find model
    model_path = find_model()
    if not model_path:
        return
    
    # On Jetson, try to use TensorRT
    if IS_JETSON and model_path.endswith('.pt'):
        model_path = export_tensorrt(model_path)
    
    print(f"[INFO] Model: {model_path}")
    
    # Load model
    from ultralytics import YOLO
    model = YOLO(model_path)
    
    # GPU setup
    if IS_JETSON:
        try:
            model.to('cuda')
            print("[INFO] GPU Jetson Nano (CUDA) đã kích hoạt.")
        except:
            print("[WARNING] GPU không khả dụng, chạy CPU.")
    else:
        print("[INFO] Chạy trên CPU (Desktop mode).")
    
    # Setup GPIO
    gpio = setup_gpio()
    
    # Open camera
    cap = open_camera()
    if cap is None:
        return
    
    print(f"[INFO] Vùng ROI: ({ROI_X1},{ROI_Y1}) -> ({ROI_X2},{ROI_Y2})")
    print(f"[INFO] Image size: {IMGSZ}px | Confidence: {CONF_THRESHOLD}")
    print("[INFO] Nhấn 'Q' để thoát.")
    print("-" * 50)

    # FPS tracking
    fps_list = collections.deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[WARNING] Không đọc được frame!")
            break

        start_time = time.time()

        # Vẽ vùng ROI
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (ROI_X1, ROI_Y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Cắt vùng ROI
        roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        
        # Nhận diện
        results = model.predict(roi_frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]

        best_det = None
        max_area = 0
        
        for box in results.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Chuyển tọa độ ROI -> Frame gốc
            gx1, gy1 = bx1 + ROI_X1, by1 + ROI_Y1
            gx2, gy2 = bx2 + ROI_X1, by2 + ROI_Y1
            
            area = (bx2 - bx1) * (by2 - by1)
            
            # Vẽ bounding box
            color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 255)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 3)
            
            # Label với nền
            label = f"{model.names[cls_id].upper()} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (gx1, gy1 - lh - 10), (gx1 + lw, gy1), color, -1)
            cv2.putText(frame, label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            if area > max_area:
                max_area = area
                best_det = {'label': model.names[cls_id], 'conf': conf}

        # Smoothing
        detection_history.append(best_det['label'] if best_det else None)
        stable_class = get_stabilized_class(detection_history)
        
        if stable_class:
            if (time.time() - last_trigger_time) > TRIGGER_COOLDOWN:
                conf_val = best_det['conf'] if best_det else 0
                trigger_actuator(stable_class, conf_val, gpio)
                last_trigger_time = time.time()
            
            cv2.putText(frame, f"KET QUA: {stable_class.upper()}", (20, HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # FPS
        elapsed = time.time() - start_time + 0.0001
        fps_list.append(1.0 / elapsed)
        avg_fps = sum(fps_list) / len(fps_list)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Platform indicator
        platform_text = "JETSON" if IS_JETSON else "DESKTOP"
        cv2.putText(frame, platform_text, (WIDTH - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Waste Sorting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if gpio and IS_JETSON:
        try:
            gpio.cleanup()
            print("[INFO] GPIO đã cleanup.")
        except:
            pass
    
    print("[INFO] Đã thoát.")

if __name__ == "__main__":
    main()
import cv2
import torch
import time
import os
import collections
import threading
import numpy as np
import gc
from flask import Flask, Response
from ultralytics import YOLO

# --- SỬA LỖI HỆ THỐNG JETSON ---
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'

# ================== CẤU HÌNH TỐI ƯU ==================
MODEL_PATH = '../models/waste_sorter/weights/best.engine' 
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '../models/waste_sorter/weights/best.pt'

IMG_SIZE = 320          
CONF_THRESHOLD = 0.50
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 150, 100, 490, 380

# GPIO Pins
SERVO_PLASTIC_PIN = 33      
SERVO_METAL_PIN = 32        

# Hệ thống đếm rác toàn cục
counts = {'METAL': 0, 'PLASTIC': 0, 'other': 0}
COLOR_MAP = {'METAL': (0, 255, 0), 'PLASTIC': (0, 0, 255), 'other': (128, 128, 128)}

FRAME_SKIP = 2 
REQUIRED_FRAMES = 3  

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

# ================== HÀM HỖ TRỢ PHẦN CỨNG ==================
pwm_plastic = None
pwm_metal = None
GPIO_AVAILABLE = False

def init_gpio():
    global pwm_plastic, pwm_metal, GPIO_AVAILABLE
    try:
        import Jetson.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup([SERVO_PLASTIC_PIN, SERVO_METAL_PIN], GPIO.OUT)
        
        pwm_plastic = GPIO.PWM(SERVO_PLASTIC_PIN, 50)
        pwm_metal = GPIO.PWM(SERVO_METAL_PIN, 50)
        pwm_plastic.start(0)
        pwm_metal.start(0)
        GPIO_AVAILABLE = True
        print("[INFO] GPIO Initialized Successfully.")
        return GPIO
    except Exception as e:
        print(f"[ERR] GPIO Init Failed: {e}. Chạy chế độ không có phần cứng.")
        return None

def set_servo_angle(pwm, angle):
    if pwm:
        duty = angle / 18.0 + 2.0
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.4)
        pwm.ChangeDutyCycle(0) 

def trigger_servo_logic(cls):
    global counts
    if cls == "PLASTIC" and pwm_plastic:
        print("--> ĐANG GẠT PLASTIC")
        counts['PLASTIC'] += 1
        set_servo_angle(pwm_plastic, 45)
        time.sleep(0.8)
        set_servo_angle(pwm_plastic, 90)
    elif cls == "METAL" and pwm_metal:
        print("--> ĐANG GẠT METAL")
        counts['METAL'] += 1
        set_servo_angle(pwm_metal, 135)
        time.sleep(0.8)
        set_servo_angle(pwm_metal, 90)
    elif cls == "other":
        counts['other'] += 1

# ================== UI & DASHBOARD ==================

def draw_dashboard(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (230, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cv2.putText(img, "WASTE TRACKER", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(img, (15, 40), (210, 40), (255, 255, 255), 1)
    
    cv2.putText(img, f"METAL:   {counts['METAL']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['METAL'], 2)
    cv2.putText(img, f"PLASTIC: {counts['PLASTIC']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['PLASTIC'], 2)
    cv2.putText(img, f"OTHER:   {counts['other']}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['other'], 2)

@app.route("/video_feed")
def video_feed():
    def generate():
        global outputFrame, lock
        while True:
            with lock:
                if outputFrame is None: continue
                _, encodedImage = cv2.imencode(".jpg", outputFrame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ================== MAIN PROCESS ==================

def main():
    global outputFrame, lock
    GPIO_LIB = init_gpio()
    
    # Chạy Web Server (Truy cập qua IP_JETSON:5000/video_feed)
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False), daemon=True).start()

    print(f"[INFO] Nạp Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='detect')
    
    def get_video_capture():
        # Thử CSI GStreamer
        gst = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
               "nvvidconv flip-method=0 ! video/x-raw, format=BGR ! appsink drop=True")
        c = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if c.isOpened(): return c
        
        # Thử USB video0 -> video1
        for i in [0, 1, 2]:
            c = cv2.VideoCapture(i)
            if c.isOpened(): return c
        return None

    cap = get_video_capture()
    if cap is None or not cap.isOpened():
        print("[ERR] Không thể mở bất kỳ Camera nào.")
        return

    frame_count = 0
    last_act_time = 0
    history = collections.deque(maxlen=10)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            draw_dashboard(frame)
            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

            if frame_count % FRAME_SKIP == 0:
                roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                results = model.predict(source=roi_frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, half=True, verbose=False)[0]

                best_det = None
                max_area = 0
                for box in results.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    area = (bx2 - bx1) * (by2 - by1)
                    if area > 1000 and area > max_area:
                        max_area = area
                        best_det = {'box': (bx1+ROI_X1, by1+ROI_Y1, bx2+ROI_X1, by2+ROI_Y1), 
                                    'label': model.names[int(box.cls[0])].upper()}

                if best_det:
                    history.append(best_det['label'])
                    cv2.rectangle(frame, (best_det['box'][0], best_det['box'][1]), 
                                 (best_det['box'][2], best_det['box'][3]), COLOR_MAP.get(best_det['label'], (255,255,255)), 2)
                else:
                    history.append(None)

            valid_hits = [h for h in history if h and h in ['METAL', 'PLASTIC']]
            if len(valid_hits) >= REQUIRED_FRAMES:
                stable_cls = collections.Counter(valid_hits).most_common(1)[0][0]
                if (time.time() - last_act_time) > 2.5: 
                    threading.Thread(target=trigger_servo_logic, args=(stable_cls,)).start()
                    last_act_time = time.time()
                    history.clear()

            with lock:
                outputFrame = frame.copy()

            if frame_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    except KeyboardInterrupt:
        print("[INFO] Đang dừng chương trình...")

    finally:
        # Giải phóng tài nguyên
        if pwm_plastic: pwm_plastic.stop()
        if pwm_metal: pwm_metal.stop()
        if GPIO_LIB: GPIO_LIB.cleanup()
        if cap: cap.release()
        # Đã xóa cv2.destroyAllWindows() để tránh lỗi trên Jetson headless
        print("[INFO] Đã dọn dẹp xong.")

if __name__ == "__main__":
    main()
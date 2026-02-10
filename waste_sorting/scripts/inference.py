import cv2
import time
import os
from ultralytics import YOLO
import collections

# ================== CONFIG ==================
MODEL_PATH = '../models/waste_sorter/weights/best.pt' 
CONF_THRESHOLD = 0.25
WIDTH, HEIGHT = 640, 480 

# Classes: 0=plastic, 1=metal, 2=other
CLASSES = [0, 1, 2]
CLASS_NAMES = ["plastic", "metal", "other"]
CLASS_COLORS = [(0, 255, 0), (0, 165, 255), (128, 128, 128)]  # Green, Orange, Gray

# Vùng ROI - CHỈ NHẬN DIỆN TRONG VÙNG NÀY
# Format: [x1, y1, x2, y2] - tọa độ góc trên trái và góc dưới phải
# Điều chỉnh các giá trị này theo vị trí băng chuyền của bạn
ROI_X1, ROI_Y1 = 150, 100  # Góc trên trái
ROI_X2, ROI_Y2 = 490, 380  # Góc dưới phải

# Smoothing - Chống rung
HISTORY_LEN = 5
detection_history = collections.deque(maxlen=HISTORY_LEN)
last_trigger_time = 0
TRIGGER_COOLDOWN = 0.5
# ============================================

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
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Không tìm thấy model tại: {os.path.abspath(MODEL_PATH)}")
        return

    print("[INFO] Đang khởi tạo YOLOv8...")
    model = YOLO(MODEL_PATH)
    
    try:
        model.to('cuda')
        print("[INFO] Đang chạy bằng GPU (CUDA).")
    except:
        print("[WARNING] Chạy bằng CPU.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print("[INFO] Nhấn 'Q' để thoát.")
    print(f"[INFO] Vùng ROI: ({ROI_X1},{ROI_Y1}) -> ({ROI_X2},{ROI_Y2})")

    while True:
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()

        # Vẽ vùng ROI (khung vàng)
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
        cv2.putText(frame, "VUNG NHAN DIEN (ROI)", (ROI_X1, ROI_Y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Cắt vùng ROI để nhận diện
        roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        
        # Nhận diện CHỈ trong vùng ROI
        results = model.predict(roi_frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)[0]

        best_det = None
        max_area = 0
        
        for box in results.boxes:
            # Tọa độ trong ROI
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Chuyển về tọa độ frame gốc
            gx1, gy1 = bx1 + ROI_X1, by1 + ROI_Y1
            gx2, gy2 = bx2 + ROI_X1, by2 + ROI_Y1
            
            # Tính diện tích
            area = (bx2 - bx1) * (by2 - by1)
            
            # Vẽ bounding box QUANH VẬT THỂ (đậm và nổi bật)
            color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 255)
            
            # Vẽ khung dày hơn (thickness=3)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 3)
            
            # Vẽ nền cho label
            label = f"{model.names[cls_id].upper()} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (gx1, gy1 - label_h - 10), (gx1 + label_w, gy1), color, -1)  # Filled
            cv2.putText(frame, label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text
            
            if area > max_area:
                max_area = area
                best_det = {'label': model.names[cls_id], 'conf': conf}

        # Stabilize
        if best_det:
            detection_history.append(best_det['label'])
        else:
            detection_history.append(None)

        stable_class = get_stabilized_class(detection_history)
        
        if stable_class:
            if (time.time() - last_trigger_time) > TRIGGER_COOLDOWN:
                print(f">> PHAT HIEN: [{stable_class.upper()}]")
                last_trigger_time = time.time()
            
            cv2.putText(frame, f"KET QUA: {stable_class.upper()}", (20, HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # FPS
        fps = 1.0 / (time.time() - start_time + 0.0001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Waste Detection - ROI Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
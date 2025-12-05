# detector_core.py
import os
import time
import json
import socket
from pathlib import Path
from datetime import datetime
from collections import deque
from sms_sender import send_sms

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from twilio.rest import Client

EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)
UPLOADED_DIR = Path("uploaded_videos")
UPLOADED_DIR.mkdir(exist_ok=True)


MODEL_PATH = "yolov8n.pt"  
ANOMALY_THRESHOLD = 3.0
WINDOW_SECONDS = 3
FPS = 12
CONFIDENCE = 0.35
SMS_COOLDOWN = 120


TWILIO_SID = os.getenv("AC31273d16fdfbaad4ad640ffe0206dd85")
TWILIO_TOKEN = os.getenv("9865d283c6edd59967b258b48e7ea3d0")
TWILIO_FROM = os.getenv("+12184526649")
ALERT_PHONE = os.getenv("08160427720")


yolo = YOLO(MODEL_PATH)
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)
face = mp_face.FaceDetection(min_detection_confidence=0.4)

def load_seating_map(path="seating_map.json"):
    p = Path(path)
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    rows = j["rows"]; cols = j["cols"]; rolls = j["rolls"]
    src = np.array(j["homography_src"], dtype=np.float32)
    dst = np.array(j["homography_dst"], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    dst_w = int(max(dst[:,0]) - min(dst[:,0])); dst_h = int(max(dst[:,1]) - min(dst[:,1]))
    cell_w = dst_w / cols; cell_h = dst_h / rows
    return {"rows":rows, "cols":cols, "rolls":rolls, "H":H, "cell_w":cell_w, "cell_h":cell_h, "dst_w":dst_w, "dst_h":dst_h}

seating = load_seating_map()

def image_point_to_seat(xy, seating_map):
    if seating_map is None:
        return (None, None, None)
    H = seating_map["H"]
    pt = np.array([[[xy[0], xy[1]]]], dtype='float32')
    dst_pt = cv2.perspectiveTransform(pt, H)[0][0]
    dx, dy = float(dst_pt[0]), float(dst_pt[1])
    col = int(dx // seating_map["cell_w"]); row = int(dy // seating_map["cell_h"])
    if row < 0 or row >= seating_map["rows"] or col < 0 or col >= seating_map["cols"]:
        return (None, None, None)
    roll = seating_map["rolls"][row][col]
    return (row, col, roll)

def send_sms_via_twilio(body, to=None):
    to_no = to or ALERT_PHONE
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, to_no]):
        print("Twilio not configured. SMS not sent. Body:", body)
        return False
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=body, from_=TWILIO_FROM, to=to_no)
        print("SMS sent, sid:", msg.sid)
        return True
    except Exception as e:
        print("SMS failed:", e)
        return False


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

last_alert_for_roll = {}
def should_alert_roll(roll):
    if roll is None:
        return True
    now = time.time()
    if roll in last_alert_for_roll and now - last_alert_for_roll[roll] < SMS_COOLDOWN:
        return False
    last_alert_for_roll[roll] = now
    return True


def compute_frame_score(yolo_results, pose_landmarks, face_detections):
    score = 0.0
    if yolo_results:
        
        for res in yolo_results:
            if hasattr(res, 'boxes'):
                for box in res.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                
                    if cls == 67:
                        score += 2.0 * conf
                    if cls == 84:
                        score += 0.8 * conf
   
    if pose_landmarks:
        try:
            lm = pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE]
            left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_center_x = (left_sh.x + right_sh.x)/2
            if abs(nose.x - shoulder_center_x) > 0.12:
                score += 1.0
        except Exception:
            pass
    
    if not face_detections:
        score += 0.6
    return score


def process_frame(frame, score_buf, seating_map=seating):
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
    results = yolo.predict(frame_rgb, imgsz=640, conf=CONFIDENCE, verbose=False)
   
    mp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose.process(mp_img)
    face_res = face.process(mp_img)

    score = compute_frame_score(results, pose_res.pose_landmarks, face_res.detections if face_res else None)
    score_buf.append(score)
    avg_score = sum(score_buf) / (len(score_buf) + 1e-9)

    alert_info = None
    if avg_score >= ANOMALY_THRESHOLD:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"evidence_{timestamp}.jpg"
        p = EVIDENCE_DIR / fname
        cv2.imwrite(str(p), frame)

        centroid = None
        if face_res and getattr(face_res, 'detections', None) and len(face_res.detections) > 0:
            fd = face_res.detections[0]
            bbox = fd.location_data.relative_bounding_box
            cx = int((bbox.xmin + bbox.width/2) * W)
            cy = int((bbox.ymin + bbox.height/2) * H)
            centroid = (cx, cy)
        else:
         
            largest_area = 0
            if results:
                for res in results:
                    if hasattr(res, 'boxes'):
                        for box in res.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            area = (x2-x1)*(y2-y1)
                            if area > largest_area:
                                largest_area = area
                                centroid = ((x1+x2)//2, (y1+y2)//2)

        row, col, roll = (None, None, None)
        if centroid:
            row, col, roll = image_point_to_seat(centroid, seating_map)

        if roll is None:
            seat_text = "Unknown"
        else:
            seat_text = f"Roll:{roll} Row:{row+1} Col:{col+1}"

        link = f"http://{get_local_ip()}:8501/?evidence={fname}"  
        sms_body = f"Alert: suspected malpractice. {seat_text}. Evidence: {link}"

       
        if should_alert_roll(roll):
            send_sms_via_twilio(sms_body)
        else:
        
            pass

        alert_info = {
            "image_path": str(p),
            "fname": fname,
            "seat_text": seat_text,
            "sms_body": sms_body,
            "timestamp": timestamp
        }


    cv2.putText(frame, f"score:{avg_score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame, alert_info


def process_uploaded_video(video_path, seating_map=seating):
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
    except Exception:
        fps = 0.0
    if not fps or fps <= 0:
        fps = FPS 

    buf_len = max(1, int(WINDOW_SECONDS * fps))
    score_buf = deque(maxlen=buf_len)

    alerts = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        processed_frame, alert = process_frame(frame, score_buf, seating_map)
        if alert:
            alerts.append(alert)
    cap.release()
    return alerts


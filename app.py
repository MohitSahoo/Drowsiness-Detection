import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import pygame
import os

# --- CONFIG ---
MODEL_PATH = 'yolov8n.pt'
ALERT_SOUND_PATH = 'alert.wav'  # Place alert.wav in the project root
CALIBRATION_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.5
DROWSY_CONSEC_FRAMES = 15
ALERT_COOLDOWN = 3
BLINK_THRESHOLD = 0.3
CONFIDENCE_WINDOW = 10
HEAD_NOD_THRESHOLD = 0.65
FACE_TILT_THRESHOLD = 0.6
MOVEMENT_THRESHOLD = 0.2
EYE_CLOSURE_THRESHOLD = 0.6

# --- Drowsiness Detector Class ---
class DrowsinessDetector:
    def __init__(self):
        self.last_positions = deque(maxlen=10)
        self.last_angles = deque(maxlen=10)
        self.movement_history = deque(maxlen=30)
        self.eye_state_history = deque(maxlen=5)
        self.last_main_box = None

    def calculate_face_angle(self, face_box):
        x1, y1, x2, y2 = face_box
        width = x2 - x1
        height = y2 - y1
        return width / height if height != 0 else 1.0

    def calculate_movement(self, current_pos, last_pos):
        if not last_pos:
            return 0
        return np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)

    def is_drowsy(self, face_box, frame_shape):
        x1, y1, x2, y2 = face_box
        height, width = frame_shape[:2]
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        face_width = x2 - x1
        face_height = y2 - y1
        current_angle = self.calculate_face_angle(face_box)
        self.last_positions.append((face_center_x, face_center_y))
        self.last_angles.append(current_angle)
        if len(self.last_positions) > 1:
            movement = self.calculate_movement(
                self.last_positions[-1],
                self.last_positions[-2]
            )
            self.movement_history.append(movement)
        drowsiness_score = 0
        drowsiness_factors = []
        # 1. Head nodding
        if len(self.last_positions) > 3:
            recent_positions = list(self.last_positions)[-3:]
            avg_y = np.mean([p[1] for p in recent_positions])
            if avg_y > height * HEAD_NOD_THRESHOLD:
                if any(p[1] > height * HEAD_NOD_THRESHOLD for p in recent_positions[-2:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("head_nodding")
        # 2. Face tilt
        if len(self.last_angles) > 3:
            recent_angles = list(self.last_angles)[-3:]
            angle_variation = np.std(recent_angles)
            if angle_variation > FACE_TILT_THRESHOLD:
                if any(abs(a - np.mean(recent_angles)) > FACE_TILT_THRESHOLD/2 for a in recent_angles[-2:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("face_tilting")
        # 3. Low movement
        if len(self.movement_history) > 5:
            recent_movement = list(self.movement_history)[-5:]
            avg_movement = np.mean(recent_movement)
            if avg_movement < MOVEMENT_THRESHOLD:
                if any(m < MOVEMENT_THRESHOLD * 1.2 for m in recent_movement[-3:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("reduced_movement")
        # 4. Simulated eye closure
        if len(self.eye_state_history) > 2:
            recent_eye_states = list(self.eye_state_history)[-2:]
            if face_height < height * EYE_CLOSURE_THRESHOLD:
                if any(state > 0 for state in recent_eye_states):
                    drowsiness_score += 1
                    drowsiness_factors.append("possible_eye_closure")
        self.eye_state_history.append(drowsiness_score)
        is_drowsy = drowsiness_score >= 2 and len(set(drowsiness_factors)) >= 2
        return is_drowsy, drowsiness_factors, drowsiness_score

# --- Helper Functions ---
def get_main_person(boxes, frame_shape):
    main_box = None
    max_score = -1
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            center_y = (y1 + y2) / 2
            position_score = 1 - abs(center_y - frame_shape[0]/2) / (frame_shape[0]/2)
            score = area * conf * (0.7 + 0.3 * position_score)
            if score > max_score:
                max_score = score
                main_box = (x1, y1, x2, y2, conf)
    return main_box

def calibrate_detection(model, cap, num_frames=CALIBRATION_FRAMES):
    st.info("Calibrating... Please look straight ahead and move normally.")
    detector = DrowsinessDetector()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detector.calculate_face_angle((x1, y1, x2, y2))
    st.success("Calibration complete. Starting drowsiness detection.")
    return detector

# --- Streamlit UI ---
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Real-Time Multi-Factor Drowsiness Detection (YOLOv8 + Custom Logic)")
st.markdown("""
- This app uses your webcam and a trained YOLOv8n model to detect a person and apply custom drowsiness logic.<br>
- Drowsiness is detected using head nodding, face tilt, low movement, and simulated eye closure.<br>
- If you see a big red warning, drowsiness has been detected!
- Make sure your webcam is connected and allowed in your browser.
- Place yolov8n.pt and alert.wav in the app directory.
""", unsafe_allow_html=True)

conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.01)
st.sidebar.info("Adjust the YOLO confidence threshold for person detection.")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_model()

# Initialize pygame for sound
if not pygame.mixer.get_init():
    pygame.mixer.init()
if os.path.exists(ALERT_SOUND_PATH):
    alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
else:
    alert_sound = None
    st.warning("alert.wav not found. No sound will play on drowsiness alert.")

run = st.button("Start Webcam Detection")
stop = st.button("Stop Webcam")
frame_placeholder = st.empty()
warning_placeholder = st.empty()
status_placeholder = st.empty()

if run and model is not None:
    cap = cv2.VideoCapture(0)
    st.info("Webcam started. Calibrating...")
    detector = calibrate_detection(model, cap)
    COUNTER = 0
    ALARM_ON = False
    last_alert_time = 0
    blink_counter = 0
    last_blink_time = 0
    confidence_buffer = deque(maxlen=CONFIDENCE_WINDOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame from webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, conf=conf_threshold)
        boxes = results[0].boxes
        main_person = get_main_person(boxes, frame.shape) if boxes is not None else None
        drowsy_detected = False
        drowsiness_factors = []
        drowsiness_score = 0
        current_confidence = 0
        if main_person:
            x1, y1, x2, y2, conf = main_person
            is_drowsy, factors, score = detector.is_drowsy((x1, y1, x2, y2), frame.shape)
            if is_drowsy:
                drowsy_detected = True
                current_confidence = conf
                drowsiness_factors = factors
                drowsiness_score = score
            # Draw box
            color = (0, 0, 255) if drowsy_detected else (0, 255, 0)
            label = f"{'Drowsy' if drowsy_detected else 'Awake'}"
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Drowsiness alert logic
        avg_confidence = np.mean(confidence_buffer) if confidence_buffer else 0
        current_time = time.time()
        if drowsy_detected:
            COUNTER += 1
            confidence_buffer.append(current_confidence)
            if current_time - last_blink_time > BLINK_THRESHOLD:
                blink_counter += 1
                last_blink_time = current_time
            if COUNTER >= DROWSY_CONSEC_FRAMES and current_time - last_alert_time > ALERT_COOLDOWN:
                if not ALARM_ON:
                    ALARM_ON = True
                    warning_placeholder.markdown('<h1 style="color:red; text-align:center;">Drowsy Detected!</h1>', unsafe_allow_html=True)
                    if alert_sound:
                        alert_sound.play()
                    last_alert_time = current_time
        else:
            COUNTER = max(0, COUNTER - 1)
            ALARM_ON = False
            warning_placeholder.empty()
        # Show frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        # Show status
        status_placeholder.markdown(f"""
        <b>Status:</b> {'<span style=\"color:red\">Drowsy</span>' if drowsy_detected else '<span style=\"color:green\">Awake</span>'}<br>
        <b>Blink Count:</b> {blink_counter}<br>
        <b>Avg Confidence:</b> {avg_confidence:.2f}<br>
        <b>Drowsiness Score:</b> {drowsiness_score}<br>
        <b>Factors:</b> {', '.join(drowsiness_factors) if drowsiness_factors else 'None'}
        """, unsafe_allow_html=True)
        if stop:
            break
        time.sleep(0.03)
    cap.release()
    frame_placeholder.empty()
    warning_placeholder.empty()
    status_placeholder.empty()
    st.success("Webcam stopped.")
elif model is None:
    st.error("Model could not be loaded. Please check the model file path and try again.")
else:
    st.info("Click 'Start Webcam Detection' to begin.")
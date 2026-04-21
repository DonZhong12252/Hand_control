#!/usr/bin/env python3
import cv2
import mediapipe as mp
import pyautogui
import time
import os
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
CAMERA_INDEX = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

latest_result = None
result_timestamp = 0

def result_callback(result, output_image, timestamp_ms):
    global latest_result, result_timestamp
    latest_result = result
    result_timestamp = timestamp_ms

def get_finger_states(landmarks):
    lm = landmarks
    return {
        'thumb':  lm[4].x < lm[3].x,
        'index':  lm[8].y < lm[6].y,
        'middle': lm[12].y < lm[10].y,
        'ring':   lm[16].y < lm[14].y,
        'pinky':  lm[20].y < lm[18].y,
    }

def detect_gesture(fingers):
    up = {f for f, v in fingers.items() if v}
    if up == {'index'}:
        return 'one_finger'
    elif up == {'index', 'middle'}:
        return 'two_fingers'
    elif up == {'index', 'middle', 'ring'}:
        return 'three_fingers'
    elif up == {'index', 'middle', 'ring', 'pinky'}:
        return 'four_fingers'
    elif up in [{'thumb', 'index'}, {'thumb'}]:
        return 'pinch'
    return 'none'

def execute_command(gesture):
    try:
        if gesture == 'one_finger':
            pyautogui.click()
            print("  -> left click")
        elif gesture == 'two_fingers':
            pyautogui.scroll(100)
            print("  -> scroll up")
        elif gesture == 'three_fingers':
            pyautogui.scroll(-100)
            print("  -> scroll down")
        elif gesture == 'pinch':
            fname = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pyautogui.screenshot().save(fname)
            print(f"  -> screenshot saved: {fname}")
    except Exception as e:
        print(f"  command error: {e}")

def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        ]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 100), -1)
    return frame

def main():
    global latest_result

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Run: curl -o ~/hand_control/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Failed to open camera. Check System Settings > Privacy > Camera.")
        return

    print("Hand tracking started. Press 'q' to quit.")
    print("Gestures: 1 finger=click  2=scroll up  3=scroll down  pinch=screenshot")

    last_gesture = 'none'
    last_command_time = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            timestamp_ms = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_result and latest_result.hand_landmarks:
                frame = draw_landmarks(frame, latest_result.hand_landmarks)
                fingers = get_finger_states(latest_result.hand_landmarks[0])
                gesture = detect_gesture(fingers)

                now = time.time()
                if gesture != 'none' and gesture != last_gesture and (now - last_command_time) > 0.6:
                    print(f"Gesture: {gesture}")
                    execute_command(gesture)
                    last_command_time = now

                last_gesture = gesture
            else:
                last_gesture = 'none'

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Hand Control

Control your Mac with hand gestures in front of your webcam. Uses
[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
to track one hand and [PyAutoGUI](https://pyautogui.readthedocs.io/) to send clicks, scrolls and screenshots.

## Gestures

| Gesture | Fingers up | Action |
|---|---|---|
| One finger | index | Left click |
| Two fingers | index + middle | Scroll up |
| Three fingers | index + middle + ring | Scroll down |
| Pinch | thumb (or thumb + index) | Save a screenshot to the current folder |

A gesture fires once when it changes, with a 0.6 s cooldown, so holding a pose doesn't repeat it.
Four fingers is recognised but not mapped to an action yet.

## Setup

```bash
git clone https://github.com/DonZhong12252/Hand_control.git
cd Hand_control
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 hand_control.py
```

The model file `hand_landmarker.task` is included. If it's missing, download it next to the script:

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### macOS permissions

- **Camera** — allow your terminal app in System Settings → Privacy & Security → Camera.
- **Accessibility** — allow your terminal app in System Settings → Privacy & Security → Accessibility, so clicks and scrolls work.
- **Screen Recording** — needed for the pinch screenshot.

Press `q` in the preview window to quit. Change `CAMERA_INDEX` in `hand_control.py` if the wrong camera opens.

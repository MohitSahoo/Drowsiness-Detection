# Drowsiness Detection with YOLOv8 and Streamlit

This project is a real-time drowsiness detection system using a YOLOv8 model and a custom multi-factor logic, all wrapped in an easy-to-use Streamlit web app. The app uses your webcam to detect if a person is drowsy or awake, based on head nodding, face tilt, low movement, and simulated eye closure.

## Features

- **Real-time webcam detection** using YOLOv8
- **Custom drowsiness logic** (not just class labels)
- **Calibration phase** for personalized detection
- **Visual alerts** and on-screen status
- **Audio buzzer alert** (requires `alert.wav` in the project root)
- **Configurable detection thresholds**
- **Robust to full/upper body in frame**

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Streamlit
- OpenCV
- Pillow
- pygame (for sound alerts)

Install all requirements with:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/MohitSahoo/Drowsiness-Detection.git
   cd Drowsiness-Detection
   ```
2. Place your trained YOLOv8 model file as `yolov8n.pt` in the project root.
3. Place an alert sound file named `alert.wav` in the project root (optional, for buzzer alert).

## Usage

Run the Streamlit app with:

```bash
streamlit run app.py
```

- Allow webcam access in your browser.
- Follow the calibration instructions.
- Adjust detection thresholds in the sidebar for best results.
- The app will show a red warning and play a sound if drowsiness is detected.

## Notes

- The app is robust to cases where the upper body is visible, and prioritizes face/head for detection.
- If the face is too small or not detected, a warning will be shown.
- All detection logic is custom and does not rely solely on YOLO class labels.

## Credits

- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Streamlit for the web UI
- Developed by Mohit Sahoo

---

Feel free to open issues or pull requests for improvements!

# 🖐️ Gesture Mouse Control

A real-time hand gesture-based mouse control system using computer vision and MediaPipe. Control your computer's mouse cursor, click, drag, and right-click using only hand gestures captured through your webcam.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **Cursor Movement**: Control mouse cursor by pointing with your index finger
- **Left Click/Drag**: Pinch thumb and index finger together to click or drag
- **Right Click**: Pinch thumb and middle finger together for right-click
- **Smooth Tracking**: Optimized performance with exponential moving average smoothing
- **Real-time Visualization**: Optional visual feedback showing hand landmarks and gesture detection
- **Adjustable Settings**: Live tuning of smoothing and sensitivity while running

## 🎥 Demo

Move your hand in front of your webcam and control your computer entirely through gestures!

**Gestures:**
- 👆 **Move Cursor**: Point with index finger
- 🤏 **Click/Drag**: Pinch thumb + index finger
- 🤞 **Right-Click**: Pinch thumb + middle finger

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Mac/Linux

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/gesture-mouse-control.git
   cd gesture-mouse-control
```

2. **Install dependencies**
```bash
   pip install opencv-python mediapipe pyautogui
```

3. **Download the MediaPipe hand tracking model**
```bash
   python download_model.py
```

## 📖 Usage

### Basic Usage

Run the main script:
```bash
python gesture_mouse_control.py
```

### Keyboard Controls

While the program is running, you can use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `v` | Toggle visualization (show/hide hand landmarks) |
| `+` | Increase alpha (faster response, less smooth) |
| `-` | Decrease alpha (slower response, more smooth) |
| `q` | Quit the application |

### Adjusting Performance

The `alpha` parameter controls the smoothing:
- **Lower values (0.2-0.3)**: Smoother cursor movement, slower response
- **Higher values (0.4-0.5)**: Faster response, less smooth movement
- **Default**: 0.45

You can adjust this in real-time using the `+` and `-` keys.

## 🛠️ Configuration

You can customize the behavior by editing these parameters in `gesture_mouse_control.py`:
```python
# Smoothing (0.1 = very smooth, 1.0 = no smoothing)
alpha = 0.45

# Dead zone at frame edges (prevents cursor jumping)
frame_reduction = 0.1

# Gesture sensitivity thresholds
pinch_threshold = 0.05        # Thumb + Index click
right_click_threshold = 0.05  # Thumb + Middle right-click
```

## 📁 Project Structure
```
gesture-mouse-control/
├── model
├    ├──download_model.py      # Script to download MediaPipe model
├    └── hand_landmarker.task  # MediaPipe hand tracking model (downloaded)
├── mouse_controller.py        # Main application
├── fast_mouse_controller.y    # Faster Optimized Main application
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🔧 Technical Details

### Technologies Used

- **OpenCV**: Webcam capture and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **PyAutoGUI**: Mouse control automation

### How It Works

1. **Hand Detection**: MediaPipe detects 21 landmarks on your hand in real-time
2. **Gesture Recognition**: Calculates distances between key landmarks (thumb, index, middle finger)
3. **Coordinate Mapping**: Converts normalized hand coordinates (0-1) to screen coordinates
4. **Smoothing**: Applies exponential moving average to reduce jitter
5. **Action Execution**: Triggers mouse movements and clicks based on detected gestures

### Performance Optimizations

- **VIDEO mode**: Uses temporal information for smoother tracking
- **Lower resolution**: Processes 640x480 frames for faster performance
- **Minimal drawing**: Optional visualization to maximize FPS
- **EMA smoothing**: Efficient smoothing algorithm with minimal computational cost
- **Dead zones**: Prevents cursor jumping when hand approaches frame edges

Expected performance: **25-35 FPS** on most modern computers

## 🎯 Use Cases

- **Accessibility**: Hands-free computer control for users with limited mobility
- **Presentations**: Control slides without a remote
- **Gaming**: Alternative input method for certain games
- **Education**: Learn computer vision and gesture recognition
- **Prototyping**: Base for more complex gesture-based interfaces

## 🐛 Troubleshooting

### Low FPS (< 15 FPS)

- Turn off visualization by pressing `v`
- Lower webcam resolution in the code
- Close other resource-intensive applications
- Try the lite model (see Configuration)

### Cursor is Jittery

- Decrease `alpha` value by pressing `-`
- Increase `frame_reduction` to create larger dead zones
- Ensure good lighting conditions

### Gestures Not Detected

- Ensure hand is clearly visible to camera
- Adjust `pinch_threshold` values
- Check if lighting is adequate
- Try increasing confidence thresholds

### Cursor Not Moving

- Check if PyAutoGUI failsafe is triggered (move mouse to corner manually)
- Verify webcam is working properly
- Ensure hand is within frame boundaries

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision tools
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project. For production use, consider adding error handling, security features, and accessibility options.

Made with ❤️ and Python
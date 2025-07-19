# 👁️ Blind Assistance

## 📌 Overview

Blind Assistance is a real-time object detection and text-to-speech system designed to assist visually impaired individuals. The project uses **YOLOv8** for object detection and **pyttsx3** for speech synthesis to announce detected objects with their directional position. 🎤

## ✨ Features

- 🔍 Real-time object detection using YOLOv8 (nano model for performance).
- 🗣️ Converts detected objects into speech with directional guidance (left, center, right).
- 📷 Uses OpenCV for camera integration and visualization.
- 🎯 Supports multiple object detection with confidence scores.
- 🕐 Intelligent speech throttling to avoid repetitive announcements.
- 🎛️ Optimized performance with frame skipping and reduced resolution.
- 🧵 Multi-threaded text-to-speech for smooth operation.

## ⚙️ Requirements

Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy pyttsx3 ultralytics torch torchvision Pillow
```

The YOLOv8 nano model (`yolov8n.pt`) will be automatically downloaded when you first run the script.

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vikranth2711/blind-assistance.git
   cd blind-assistance
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python project-final.py
   ```

## 🎥 How It Works

1. 📸 The camera captures real-time video frames at 640x480 resolution.
2. 🧠 YOLOv8 detects objects with a minimum confidence threshold of 0.5.
3. 📍 The system determines object position (left, center, right) based on bounding box center.
4. 🖼️ Bounding boxes and labels with directional info are drawn on detected objects.
5. 🔊 Objects are announced with their direction (e.g., "person on the left").
6. ⏱️ Speech throttling prevents repeated announcements of the same object within 5 seconds.
7. 🔄 The system processes every 2nd frame for optimal performance.
8. ❌ Press 'q' to exit the application.

## 🎛️ Configuration Options

You can modify these constants in the code:

- `FRAME_SKIP = 2`: Process every nth frame (higher = better performance, lower accuracy)
- `FRAME_WIDTH, FRAME_HEIGHT = 640, 480`: Camera resolution
- `CONFIDENCE_THRESHOLD = 0.5`: Minimum detection confidence (0.0-1.0)
- `SPEECH_INTERVAL = 5`: Seconds between same object announcements
- Voice settings: Rate (150 WPM) and volume (0.9)

## 🔧 Technical Details

- **Model**: YOLOv8n (nano) - optimized for real-time inference
- **Framework**: Ultralytics YOLO
- **Dataset**: Pre-trained on COCO dataset (80+ object classes)
- **Performance**: ~30 FPS on modern hardware with frame skipping
- **Memory**: Low memory footprint with nano model

## 📋 Supported Objects

The system can detect 80+ object classes from the COCO dataset including:

- People, vehicles, animals
- Furniture, electronics, kitchen items
- Sports equipment, tools, and more

## 🔮 Future Improvements

- ⚡ Support for YOLOv8s/m/l models for better accuracy.
- 📏 Add distance estimation using depth cameras or stereo vision.
- 📱 Mobile app version with camera integration.
- 🗣️ Enhanced speech with object descriptions and contextual information.
- 🎯 Custom training for specific environments or objects.
- 🔊 Audio cues and spatial audio for better directional guidance.
- ⚙️ GUI configuration panel for real-time parameter adjustment.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. 💡

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision tools
- [pyttsx3](https://pypi.org/project/pyttsx3/) for text-to-speech functionality

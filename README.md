# 👁️ Blind Assistance

## 📌 Overview
Blind Assistance is a real-time object detection and text-to-speech system designed to assist visually impaired individuals. The project uses **YOLOv3** for object detection and **pyttsx3** for speech synthesis to announce detected objects. 🎤

## ✨ Features
- 🔍 Real-time object detection using YOLOv3.
- 🗣️ Converts detected objects into speech for user assistance.
- 📷 Uses OpenCV for camera integration and visualization.
- 🎯 Supports multiple object detection with confidence scores.

## ⚙️ Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy pyttsx3
```

You also need to download the **YOLOv3 weights and configuration files**:
- [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights) 📥
- [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) ⚙️
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names) 📜

## 🚀 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vikranth2711/blind-assistance.git
   cd blind-assistance
   ```
2. Place the YOLOv3 model files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`) inside the project directory.
3. Run the script:
   ```bash
   python blind_assistance.py
   ```

## 🎥 How It Works
1. 📸 The camera captures real-time video frames.
2. 🧠 YOLOv3 detects objects and assigns confidence scores.
3. 🖼️ The system draws bounding boxes and labels on detected objects.
4. 🔊 The detected object's name is converted to speech using **pyttsx3**.
5. 🔄 The system continuously updates detections until terminated (`q` key).

## 🔮 Future Improvements
- ⚡ Improve performance with **YOLOv4-tiny** or **YOLOv5**.
- 📏 Add distance estimation for better object localization.
- 📱 Implement mobile or Raspberry Pi support.
- 🗣️ Enhance speech output with contextual descriptions.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests. 💡


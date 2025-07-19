import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from threading import Thread
import time

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Nano model for real-time performance

# Initialize text-to-speech engine
voice = pyttsx3.init()
voice.setProperty('rate', 150)  # Speed of speech
voice.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Set frame processing frequency and resolution
FRAME_SKIP = 2  # Process every 2nd frame
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # Reduced resolution for performance
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
SPEECH_INTERVAL = 5  # Seconds between same object announcements

# Dictionary to track last spoken time for each object
last_spoken = {}

# Initialize frame counter
frame_count = 0

def speak_objects(objects):
    """Run text-to-speech in a separate thread."""
    for obj, direction in objects:
        voice.say(f"{obj} on the {direction}")
    voice.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    # Resize frame for faster processing
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Process only every FRAME_SKIP-th frame
    if frame_count % FRAME_SKIP == 0:
        # Perform YOLOv8 inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]

        detected_objects = []  # Store (object, direction) tuples
        # Iterate over detections
        for box in results.boxes:
            # Get bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            obj_name = results.names[class_id]
            label = f"{obj_name}: {confidence:.2f}"

            # Determine direction based on bounding box center
            center_x = (x1 + x2) // 2
            frame_center = FRAME_WIDTH // 2
            direction = (
                "left" if center_x < frame_center - 50
                else "right" if center_x > frame_center + 50
                else "center"
            )

            # Draw bounding box and label
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({direction})", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add object to list if not recently spoken
            if obj_name not in last_spoken or (time.time() - last_spoken[obj_name]) > SPEECH_INTERVAL:
                detected_objects.append((obj_name, direction))
                last_spoken[obj_name] = time.time()

        # Run text-to-speech in a separate thread if objects detected
        if detected_objects:
            Thread(target=speak_objects, args=(detected_objects,)).start()

        print("Detected:", ", ".join([f"{obj} ({dir})" for obj, dir in detected_objects]))

    frame_count += 1  # Increment frame count

    # Display the output frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
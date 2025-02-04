import cv2
import numpy as np
import pyttsx3

# ====================== FILE PATHS (REPLACE WITH YOURS) ====================== #
# Replace these paths with the actual locations where you downloaded the files
cfg = r"PATH_TO_YOLOV3_CFG\yolov3.cfg"  # Example: r"C:\Users\YourName\Downloads\yolov3.cfg"
weights = r"PATH_TO_YOLOV3_WEIGHTS\yolov3.weights"  # Example: r"C:\Users\YourName\Downloads\yolov3.weights"
coco_names = r"PATH_TO_COCO_NAMES\coco.names"  # Example: r"C:\Users\YourName\Downloads\coco.names"
# ============================================================================ #

# Load YOLO model
net = cv2.dnn.readNet(weights, cfg)

# Load class labels from coco.names
classes = []
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set thresholds for object detection
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for a detection
NMS_THRESHOLD = 0.4  # Non-Maximum Suppression threshold

# Initialize text-to-speech engine
voice = pyttsx3.init()

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Retrieve YOLO output layer names (ensures compatibility with different OpenCV versions)
layer_names = net.getLayerNames()
try:
    output_layers = net.getUnconnectedOutLayersNames()
except AttributeError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set frame processing frequency (skip frames for better performance)
FRAME_SKIP = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    # Process only every FRAME_SKIP-th frame
    if frame_count % FRAME_SKIP == 0:
        height, width = frame.shape[:2]

        # Convert frame to blob (YOLO format)
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        # Run forward pass through YOLO
        outs = net.forward(output_layers)

        class_ids = []  # Store detected object class IDs
        confidences = []  # Store confidence scores
        boxes = []  # Store bounding box coordinates

        # Iterate over YOLO outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]  # Class confidence scores
                class_id = np.argmax(scores)  # Get highest scoring class
                confidence = scores[class_id]  # Get confidence value

                if confidence > CONFIDENCE_THRESHOLD:
                    # Get bounding box coordinates
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to reduce redundant boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Draw bounding boxes and labels
        detected_objects = set()
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green color for bounding box

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Add detected object to the set
                detected_objects.add(classes[class_ids[i]])

        # Convert detected object names to speech
        for obj in detected_objects:
            voice.say(obj)  # Speak detected object name
        voice.runAndWait()

        print("Detected:", ", ".join(detected_objects))

    frame_count += 1  # Increment frame count

    # Display the output frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

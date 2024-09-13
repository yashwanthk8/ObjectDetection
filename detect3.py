import cv2
import numpy as np
import os
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the camera feed
cap = cv2.VideoCapture(0)

# Settings
frame_skip = 1  # Process every 2nd frame
input_size = (320, 320)  # Smaller input size for faster processing
last_beep_time = 0  # Track the time of the last beep
beep_cooldown = 2  # Cooldown time in seconds

def play_beep():
    os.system('afplay /System/Library/Sounds/Glass.aiff')

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Skip frames
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    height, width, channels = frame.shape

    # Preprocess the input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, input_size, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extracting information from the detections
    class_ids = []
    confidences = []
    boxes = []
    object_detected = False

    for out in outs:
        for detection in out:
            # Ensure that the detection has the right number of elements
            if len(detection) >= 5:
                # Extract bounding box coordinates and class probabilities
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordinates for bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Get the confidence scores and class ID
                scores = detection[5:]  # The last part of detection contains the class scores
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    object_detected = True

                    # Object detected
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Perform Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Play beep if an object is detected and cooldown is over
    current_time = time.time()
    if object_detected and current_time - last_beep_time > beep_cooldown:
        play_beep()
        last_beep_time = current_time  # Reset the cooldown timer

    # Show the frame
    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()

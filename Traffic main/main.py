import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Load YOLO model
model = YOLO('best.pt')

# Load class names
with open("coco1.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Mouse hover function to show coordinates
def show_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse at: ({x}, {y})")

# Setup display window and attach mouse callback
cv2.namedWindow('Detection')
cv2.setMouseCallback('Detection', show_mouse_position)

# Open video file
cap = cv2.VideoCapture('he2.mp4')

# Frame processing settings
frame_skip = 3  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frame for better visibility
    frame = cv2.resize(frame, (1020, 500))

    # YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data

    if detections is not None and len(detections) > 0:
        detections_df = pd.DataFrame(detections.cpu().numpy()).astype("float")

        for _, row in detections_df.iterrows():
            x1, y1, x2, y2, confidence, class_id = map(int, row[:6])
            class_name = class_list[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Semi-transparent background for label
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1 - 30), (x2, y1), (255, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Display class label and confidence score
            cvzone.putTextRect(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 5), scale=1, thickness=1)

            # Print detected object
            print(f"Detected: {class_name} with {confidence:.2f} confidence.")

    # Show frame
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

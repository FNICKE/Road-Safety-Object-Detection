import cv2
from camera import Camera
from detection import ObjectDetector
from speed_detection import SpeedDetection
from driver_attention import DriverAttention
from utils import send_email, send_sms, calculate_distance

def main():
    # Initialize video capture (camera or recorded video)
    camera = Camera(source=0)  # 0 for webcam, or use file path for recorded video

    # Load Object Detector (YOLO or other models)
    detector = ObjectDetector(config_path='resources/yolov3.cfg', 
                               weights_path='resources/yolov3.weights', 
                               class_names_path='resources/coco.names')

    # Initialize Speed Detection
    speed_detector = SpeedDetection()

    # Initialize Driver Attention Monitoring
    attention_monitor = DriverAttention(cascade_path='resources/haarcascades/haarcascade_frontalface_default.xml')

    while True:
        frame = camera.get_frame()
        
        if frame is None:
            break

        # Perform object detection (vehicles, pedestrians, etc.)
        detections = detector.detect(frame)

        # Perform speed detection if vehicle is detected
        speed = speed_detector.detect_speed(detections)

        # Check driver attention (e.g., driver looking at the road or distracted)
        attention_status = attention_monitor.check_attention(frame)

        # If attention is not detected, send an alert
        if attention_status == 'distracted':
            send_email("Driver Attention Alert", "The driver is distracted", to_email="alert@example.com")
            send_sms("The driver is distracted", to_phone="+1234567890")

        # Display the frame with detections and alerts
        cv2.putText(frame, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Driver Attention: {attention_status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Road Safety Monitor", frame)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

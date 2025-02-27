import cv2

class Camera:
    def __init__(self, source=0):
        # source: 0 for webcam, or path for recorded video
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

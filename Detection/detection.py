import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, config_path, weights_path, class_names_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_layers)

        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    detections.append((x, y, w, h, class_id, confidence))
        
        return detections

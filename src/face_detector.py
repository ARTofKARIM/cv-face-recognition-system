"""Face detection using Haar cascades and DNN."""
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, method="haar", min_face_size=30, confidence=0.7):
        self.method = method
        self.min_face_size = min_face_size
        self.confidence = confidence
        if method == "haar":
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, image):
        if self.method == "haar":
            return self._detect_haar(image)
        return []

    def _detect_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                minSize=(self.min_face_size, self.min_face_size))
        return [(x, y, w, h) for (x, y, w, h) in faces]

    def extract_faces(self, image, target_size=(160, 160)):
        boxes = self.detect(image)
        faces = []
        for (x, y, w, h) in boxes:
            margin = int(0.1 * w)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, target_size)
            faces.append({"image": face, "box": (x, y, w, h)})
        return faces

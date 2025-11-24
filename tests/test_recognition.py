"""Tests for face recognition."""
import unittest
import numpy as np
from src.face_database import FaceDatabase
from src.recognizer import FaceRecognizer

class TestFaceDatabase(unittest.TestCase):
    def test_add_and_size(self):
        db = FaceDatabase()
        db.add_face("Alice", np.random.randn(128))
        db.add_face("Bob", np.random.randn(128))
        self.assertEqual(db.size(), 2)

class TestRecognizer(unittest.TestCase):
    def test_recognize_known(self):
        db = FaceDatabase()
        emb = np.random.randn(128)
        db.add_face("Alice", emb)
        rec = FaceRecognizer(db, threshold=0.5, metric="cosine")
        name, dist = rec.recognize(emb + np.random.randn(128) * 0.01)
        self.assertEqual(name, "Alice")

if __name__ == "__main__":
    unittest.main()

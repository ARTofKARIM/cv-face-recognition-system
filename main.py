"""Main entry point for face recognition."""
import argparse
from src.face_detector import FaceDetector
from src.embedding_model import FaceEmbeddingModel
from src.face_database import FaceDatabase
from src.recognizer import FaceRecognizer

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--image", help="Image to process")
    parser.add_argument("--database", help="Path to face database JSON")
    parser.add_argument("--enroll", help="Enroll face with name")
    args = parser.parse_args()

    detector = FaceDetector()
    model = FaceEmbeddingModel()
    model.build()
    db = FaceDatabase()
    if args.database:
        db.load(args.database)
    print(f"Database: {db.size()} faces, {len(db.names)} identities")

if __name__ == "__main__":
    main()

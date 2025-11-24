# Face Recognition System

A face recognition pipeline with detection, embedding generation, database management, and identity verification.

## Architecture
```
cv-face-recognition-system/
├── src/
│   ├── face_detector.py      # Haar cascade face detection
│   ├── embedding_model.py    # MobileNetV2-based face embeddings
│   ├── face_database.py      # Embedding storage and retrieval
│   ├── recognizer.py         # Cosine/Euclidean identity matching
│   └── visualization.py      # Bounding boxes, t-SNE plots
├── config/config.yaml
├── tests/test_recognition.py
└── main.py
```

## Installation
```bash
git clone https://github.com/mouachiqab/cv-face-recognition-system.git
cd cv-face-recognition-system && pip install -r requirements.txt
python main.py --image data/photo.jpg --database data/faces.json
```

## Technologies
- Python 3.9+, TensorFlow, OpenCV, dlib, scikit-learn












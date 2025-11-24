"""Face embedding database for recognition."""
import numpy as np
import json
import os

class FaceDatabase:
    def __init__(self):
        self.embeddings = {}
        self.names = []

    def add_face(self, name, embedding):
        if name not in self.embeddings:
            self.embeddings[name] = []
        self.embeddings[name].append(embedding)
        if name not in self.names:
            self.names.append(name)

    def get_all_embeddings(self):
        all_embs, all_names = [], []
        for name, embs in self.embeddings.items():
            for emb in embs:
                all_embs.append(emb)
                all_names.append(name)
        return np.array(all_embs), all_names

    def save(self, filepath):
        data = {name: [e.tolist() for e in embs] for name, embs in self.embeddings.items()}
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
        self.embeddings = {name: [np.array(e) for e in embs] for name, embs in data.items()}
        self.names = list(self.embeddings.keys())

    def size(self):
        return sum(len(v) for v in self.embeddings.values())

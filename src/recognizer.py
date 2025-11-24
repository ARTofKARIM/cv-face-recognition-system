"""Face recognition using embedding comparison."""
import numpy as np
from scipy.spatial.distance import cosine, euclidean

class FaceRecognizer:
    def __init__(self, database, threshold=0.6, metric="cosine"):
        self.database = database
        self.threshold = threshold
        self.metric = metric

    def _distance(self, emb1, emb2):
        if self.metric == "cosine":
            return cosine(emb1, emb2)
        return euclidean(emb1, emb2)

    def recognize(self, query_embedding):
        all_embs, all_names = self.database.get_all_embeddings()
        if len(all_embs) == 0:
            return "Unknown", 1.0
        distances = [self._distance(query_embedding, emb) for emb in all_embs]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        if min_dist < self.threshold:
            return all_names[min_idx], min_dist
        return "Unknown", min_dist

    def recognize_batch(self, query_embeddings):
        return [self.recognize(emb) for emb in query_embeddings]

"""Face recognition visualization."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class FaceVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def draw_results(self, image, faces, names, distances, save_path=None):
        img = image.copy()
        for face, name, dist in zip(faces, names, distances):
            x, y, w, h = face["box"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({dist:.2f})"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    def plot_embeddings_tsne(self, embeddings, labels, save=True):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            ax.scatter(coords[mask, 0], coords[mask, 1], label=label, color=colors[i], s=50)
        ax.legend()
        ax.set_title("Face Embeddings (t-SNE)")
        if save:
            fig.savefig(f"{self.output_dir}embeddings_tsne.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

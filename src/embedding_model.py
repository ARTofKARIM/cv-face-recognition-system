"""Face embedding model for generating face vectors."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FaceEmbeddingModel:
    def __init__(self, input_size=(160, 160), embedding_dim=128):
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.model = None

    def build(self):
        inputs = keras.Input(shape=(*self.input_size, 3))
        base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(*self.input_size, 3))
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.embedding_dim)(x)
        outputs = tf.math.l2_normalize(x, axis=1)
        self.model = keras.Model(inputs, outputs)
        print(f"Model built: {self.model.count_params():,} params, embedding_dim={self.embedding_dim}")
        return self.model

    def embed(self, face_image):
        if face_image.ndim == 3:
            face_image = np.expand_dims(face_image, 0)
        face_image = face_image.astype(np.float32) / 255.0
        return self.model.predict(face_image, verbose=0)[0]

    def embed_batch(self, face_images):
        batch = np.array(face_images).astype(np.float32) / 255.0
        return self.model.predict(batch, verbose=0)

"""
Epic 3
BiLSTM Emotion Classifier

Author: Sahasra
"""

from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class BiLSTMEmotionClassifier:

    def __init__(self):

        root = Path(__file__).resolve().parent.parent

        self.model = load_model(root / "bilstm_emotion.keras")

        with open(root / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(root / "label_mapping.json", "r") as f:
            self.label_mapping = json.load(f)

        self.max_length = 128

    def predict(self, text):

        sequence = self.tokenizer.texts_to_sequences([text])

        padded = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

        prediction = self.model.predict(
            padded,
            verbose=0
        )[0]

        emotion_id = int(np.argmax(prediction))

        emotion = self.label_mapping[str(emotion_id)]

        confidence = float(prediction[emotion_id])

        scores = {}

        for i, score in enumerate(prediction):
            scores[self.label_mapping[str(i)]] = float(score)

        return emotion, confidence, scores
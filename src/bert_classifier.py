"""
Epic 3
BERT Emotion Classifier

Author: Sahasra
"""

from pathlib import Path
import json

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class BertEmotionClassifier:

    def __init__(self):

        self.model_path = Path(__file__).resolve().parent.parent

        print("Loading trained BERT model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

        weights_path = self.model_path / "class_weights.json"

        with open(weights_path, "r") as f:
            weight_data = json.load(f)

        self.class_weights = weight_data["weights"]

        print("Loaded class weights:")
        print(self.class_weights)

        print("BERT model loaded successfully.\n")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]

        scores = {}

        for i, probability in enumerate(probabilities):

            label = (
                self.id2label[str(i)]
                if str(i) in self.id2label
                else self.id2label[i]
            )

            scores[label] = float(probability)

        predicted = max(scores, key=scores.get)

        confidence = scores[predicted]

        return {
            "emotion": predicted,
            "confidence": confidence,
            "scores": scores,
            "cleaned_text": text
        }
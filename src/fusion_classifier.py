"""
Epic 3
Hybrid Emotion Classifier

Author: Sahasra
"""

from src.bert_classifier import BertEmotionClassifier
from src.bilstm_classifier import BiLSTMEmotionClassifier
from src.keyword_enhancer import KeywordEnhancer


class HybridEmotionClassifier:

    def __init__(self):

        self.bert = BertEmotionClassifier()
        self.bilstm = BiLSTMEmotionClassifier()
        self.keyword = KeywordEnhancer()

    # ----------------------------------------------------
    # Convert BiLSTM emotions -> Student emotions
    # ----------------------------------------------------
    def map_bilstm_to_student(self, bilstm_scores):

        mapped = {
            "Bored": 0.0,
            "Confident": 0.0,
            "Confused": 0.0,
            "Curious": 0.0,
            "Frustrated": 0.0
        }

        mapped["Bored"] += bilstm_scores["Sadness"]
        mapped["Bored"] += bilstm_scores["Neutral"] * 0.5

        mapped["Confident"] += bilstm_scores["Joy"]
        mapped["Confident"] += bilstm_scores["Love"]

        mapped["Confused"] += bilstm_scores["Fear"]

        mapped["Curious"] += bilstm_scores["Surprise"]

        mapped["Frustrated"] += bilstm_scores["Anger"]

        return mapped

    # ----------------------------------------------------
    # Final Prediction
    # ----------------------------------------------------
    def predict(self, text):

        # BERT
        bert_result = self.bert.predict(text)
        bert_scores = bert_result["scores"]

        # BiLSTM
        _, _, bilstm_scores = self.bilstm.predict(text)

        # Map emotions
        bilstm_scores = self.map_bilstm_to_student(
            bilstm_scores
        )

        # Weighted fusion
        fused_scores = {}

        for emotion in bert_scores:

            fused_scores[emotion] = (
                0.70 * bert_scores[emotion]
                + 0.30 * bilstm_scores[emotion]
            )

        # Keyword enhancement
        enhanced_scores = self.keyword.enhance_probabilities(
            fused_scores,
            text
        )

        final_emotion = max(
            enhanced_scores,
            key=enhanced_scores.get
        )

        confidence = enhanced_scores[final_emotion]

        return final_emotion, confidence, enhanced_scores
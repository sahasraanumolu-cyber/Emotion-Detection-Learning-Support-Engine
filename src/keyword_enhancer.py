"""
Epic 3
Keyword Enhancement Module

Author: Sahasra
"""

from typing import Dict


class KeywordEnhancer:
    """
    Boosts emotion probabilities based on explicit emotion keywords.
    """

    def __init__(self):

        self.emotion_keywords = {

            "Frustrated": [
                "frustrated",
                "frustrating",
                "angry",
                "annoying",
                "hate",
                "difficult",
                "stuck",
                "wrong answer",
                "keep getting",
                "complicated"
            ],

            "Curious": [
                "why",
                "how",
                "what",
                "wonder",
                "interested",
                "learn",
                "explore",
                "know more",
                "question"
            ],

            "Confident": [
                "easy",
                "great",
                "excellent",
                "good",
                "awesome",
                "perfect",
                "solved",
                "got it",
                "clear",
                "understand"
            ],

            "Bored": [
                "boring",
                "bored",
                "repetitive",
                "dull",
                "not engaging"
            ],

            "Confused": [
                "confused",
                "unclear",
                "lost",
                "don't understand",
                "doesn't make sense",
                "missing"
            ]
        }

    def calculate_keyword_scores(self, text: str) -> Dict[str, int]:
        """
        Count keyword matches for every emotion.
        """

        text = text.lower()

        scores = {}

        for emotion, keywords in self.emotion_keywords.items():

            score = 0

            for keyword in keywords:

                if keyword in text:

                    # Strong boost for explicit emotion words
                    if keyword in [
                        "frustrated",
                        "curious",
                        "confident",
                        "bored",
                        "confused"
                    ]:
                        score += 10
                    else:
                        score += 2

            scores[emotion] = score

        return scores
    def enhance_probabilities(self, probabilities: Dict[str, float], text: str) -> Dict[str, float]:
        """
        Enhance model probabilities using keyword scores.
        """

        scores = self.calculate_keyword_scores(text)

        max_score = max(scores.values())

        # No keywords found
        if max_score == 0:
         return probabilities

        enhanced = probabilities.copy()

        # Boost emotions with the highest keyword score
        for emotion, score in scores.items():
         if score == max_score:
            enhanced[emotion] *= (1 + score / 3.0)

        # Reduce competing emotions slightly
        winners = [e for e, s in scores.items() if s == max_score]

        for emotion in enhanced:
         if emotion not in winners and max_score >= 5:
            enhanced[emotion] *= 0.5

        # Renormalize
        total = sum(enhanced.values())

        if total > 0:
          for emotion in enhanced:
            enhanced[emotion] /= total

        return enhanced
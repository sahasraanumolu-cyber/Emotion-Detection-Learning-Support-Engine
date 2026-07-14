"""
Epic 3
Mixed Emotion Detection Module

Author: Sahasra
"""

from typing import Dict, List, Tuple


class MixedEmotionDetector:
    """
    Detects multiple emotions whose confidence
    exceeds a configurable threshold.
    """

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def detect(
        self,
        scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Detect primary and secondary emotions.

        Returns:
            [
                ("Bored",0.37),
                ("Curious",0.28),
                ("Confused",0.20)
            ]
        """

        if not scores:
            return []

        # Sort emotions by confidence
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Highest probability is always included
        detected = [sorted_scores[0]]

        # Add secondary emotions above threshold
        for emotion, score in sorted_scores[1:]:

            if score >= self.threshold:
                detected.append((emotion, score))

        return detected
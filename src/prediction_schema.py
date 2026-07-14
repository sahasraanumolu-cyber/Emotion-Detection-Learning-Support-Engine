"""
Epic 3
Unified Prediction Schema

Author: Sahasra
"""

from typing import Dict, List, Tuple


class PredictionSchema:
    """
    Creates a standardized prediction output.
    """

    def build(
        self,
        emotion: str,
        confidence: float,
        scores: Dict[str, float],
        cleaned_text: str,
        mixed_emotions: List[Tuple[str, float]]
    ) -> Dict:

        return {

            "emotion": emotion,

            "confidence": confidence,

            "scores": scores,

            "mixed_emotions": mixed_emotions,

            "cleaned_text": cleaned_text

        }
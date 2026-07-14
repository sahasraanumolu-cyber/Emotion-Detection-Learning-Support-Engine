"""
Epic 3
Text Preprocessing Module

Author: Sahasra
"""

import re
from typing import List


class TextPreprocessor:
    """
    Cleans raw student text before emotion prediction.
    """

    def __init__(self):

        # Keep emotional meaning by removing only a few articles
        self.skip_words = {
            "the",
            "a",
            "an"
        }

    def clean_text(self, text: str) -> str:
        """
        Clean raw input text.
        """

        if text is None:
            return ""

        # Convert to string
        text = str(text)

        # Lowercase
        text = text.lower()

        # Remove unwanted symbols
        text = re.sub(r"[^a-zA-Z\s']", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Split cleaned text into tokens.
        """

        if not text:
            return []

        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove only basic articles.
        """

        filtered = []

        for token in tokens:

            if token not in self.skip_words and len(token) > 1:
                filtered.append(token)

        return filtered

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        """

        cleaned = self.clean_text(text)

        tokens = self.tokenize(cleaned)

        tokens = self.remove_stopwords(tokens)

        return " ".join(tokens)
"""
Epic 3
CSV Prediction Logger

Author: Sahasra
"""

import csv
from pathlib import Path
from datetime import datetime


class CSVLogger:

    def __init__(self):

        self.file_path = Path("prediction_history.csv")

        if not self.file_path.exists():

            with open(
                self.file_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Input Text",
                    "Cleaned Text",
                    "Emotion",
                    "Confidence"
                ])

    def log(
        self,
        input_text,
        cleaned_text,
        emotion,
        confidence
    ):

        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                datetime.now(),

                input_text,

                cleaned_text,

                emotion,

                round(confidence, 4)

            ])

        print("Prediction saved successfully.")
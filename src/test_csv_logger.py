from src.csv_logger import CSVLogger

logger = CSVLogger()

logger.log(

    input_text="I don't understand this topic.",

    cleaned_text="don't understand this topic",

    emotion="Confused",

    confidence=0.91

)

print("Done!")
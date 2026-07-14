from src.prediction_schema import PredictionSchema

schema = PredictionSchema()

scores = {

    "Bored": 0.3720,
    "Confident": 0.0455,
    "Confused": 0.2049,
    "Curious": 0.2772,
    "Frustrated": 0.1005

}

mixed = [

    ("Bored", 0.3720),
    ("Curious", 0.2772),
    ("Confused", 0.2049)

]

result = schema.build(

    emotion="Bored",

    confidence=0.3720,

    scores=scores,

    cleaned_text="nothing interesting is happening today",

    mixed_emotions=mixed

)

print("=" * 70)

for key, value in result.items():

    print(f"{key}:\n{value}\n")
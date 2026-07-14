from src.mixed_emotion import MixedEmotionDetector

detector = MixedEmotionDetector()

examples = [

    {
        "Bored":0.3720,
        "Confident":0.0455,
        "Confused":0.2049,
        "Curious":0.2772,
        "Frustrated":0.1005
    },

    {
        "Bored":0.04,
        "Confident":0.82,
        "Confused":0.05,
        "Curious":0.04,
        "Frustrated":0.05
    },

    {
        "Bored":0.18,
        "Confident":0.16,
        "Confused":0.15,
        "Curious":0.30,
        "Frustrated":0.21
    }

]

for i, scores in enumerate(examples, start=1):

    print("=" * 70)

    print(f"Example {i}")

    detected = detector.detect(scores)

    print("Detected Emotions:")

    for emotion, score in detected:
        print(f"{emotion:12}: {score:.2%}")
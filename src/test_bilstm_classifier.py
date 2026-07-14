from src.bilstm_classifier import BiLSTMEmotionClassifier

classifier = BiLSTMEmotionClassifier()

examples = [

    "I am really frustrated today.",

    "I finally solved the assignment!",

    "Nothing interesting is happening today.",

    "I don't understand this topic.",

    "I am excited to learn AI."

]

for text in examples:

    print("=" * 70)

    print("Text:", text)

    emotion, confidence, scores = classifier.predict(text)

    print()

    print("Emotion:", emotion)

    print("Confidence:", round(confidence * 100, 2), "%")

    print()

    print("Scores:")

    for emotion_name, score in scores.items():

        print(f"{emotion_name:<12}: {score:.4f}")
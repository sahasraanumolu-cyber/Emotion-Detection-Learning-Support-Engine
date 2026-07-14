import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_PATH = Path(__file__).resolve().parent
MAX_LENGTH = 96

print("Loading Epic 2 emotion model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Epic 2 emotion model loaded successfully.")
print("Type 'exit' to stop.\n")


def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_id = probabilities.argmax().item()

    predicted_emotion = model.config.id2label[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item() * 100

    all_probabilities = {
        model.config.id2label[index]:
        round(probability.item() * 100, 2)
        for index, probability in enumerate(probabilities)
    }

    return (
        predicted_emotion,
        confidence,
        all_probabilities,
    )


while True:
    text = input("Enter text: ").strip()

    if text.lower() == "exit":
        print("Emotion detector stopped.")
        break

    if not text:
        print("Please enter some text.\n")
        continue

    emotion, confidence, probabilities = predict_emotion(text)

    print("\nPREDICTED EMOTION:")
    print(emotion)

    print("\nCONFIDENCE:")
    print(f"{confidence:.2f}%")

    print("\nALL PROBABILITIES:")
    print(probabilities)

    print("\n" + "=" * 60 + "\n")
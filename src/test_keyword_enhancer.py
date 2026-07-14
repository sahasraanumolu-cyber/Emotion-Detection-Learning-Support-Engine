from src.keyword_enhancer import KeywordEnhancer

enhancer = KeywordEnhancer()

probabilities = {
    "Bored": 0.20,
    "Confident": 0.20,
    "Confused": 0.20,
    "Curious": 0.20,
    "Frustrated": 0.20
}

examples = [
    "I am really frustrated today",
    "Why does this happen?",
    "I finally solved it",
    "I don't understand this topic"
]

for text in examples:

    print("=" * 70)
    print("Text:", text)

    print("\nOriginal:")
    print(probabilities)

    boosted = enhancer.enhance_probabilities(probabilities, text)

    print("\nEnhanced:")
    print(boosted)

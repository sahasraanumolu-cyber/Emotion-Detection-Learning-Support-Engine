from src.preprocessing import TextPreprocessor

processor = TextPreprocessor()

examples = [

    "I am really frustrated today!!",

    "The exam was amazing.",

    "I don't understand this topic.",

    "Nothing interesting is happening today.",

    "AI is actually very fascinating!"

]

for text in examples:

    print("=" * 60)

    print("Original :", text)

    print("Processed:", processor.preprocess(text))
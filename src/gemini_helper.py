from src.response_templates import RESPONSES
import os

from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Configure Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load model
model = genai.GenerativeModel("gemini-3.5-flash")


def get_gemini_response(field, problem, emotion, confidence):

    prompt = f"""
You are an expert educational mentor.

Student Field:
{field}

Detected Emotion:
{emotion}

Confidence:
{confidence:.2%}

Problem:
{problem}

Please:
1. Acknowledge the student's emotion.
2. Explain the topic in an encouraging way.
3. Give one practical study suggestion.
4. End with motivation.
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception:

     return RESPONSES.get(
        emotion,
        "Keep learning one step at a time."
    )
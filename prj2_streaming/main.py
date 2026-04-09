from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.5-flash"

# ---

def stream_single_turn(user_msg: str, system: str, response_type: "text" or "json" = "text"):
    res = client.models.generate_content_stream(
        model=model, config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=860,
                response_mime_type="text/plain" if response_type == "text" else "application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=320),
        ),
        contents=user_msg
    ); full_text = ""
    for chunk in res:
        if chunk.text:
            print(chunk.text if response_type == "text" else chunk.text.strip(), end="", flush=True)
            full_text+=chunk.text
    return full_text if response_type == "text" else json.loads(full_text.strip())

# stream_single_turn(
#     user_msg="Define antiderivate in 3 sentences.",
#     system="You are a helpful assistant that will aid in user's requests",
#     response_type="text"
# )

schema = '{"name": string, "age": number or null, "city": string or null}'
raw_text = "My name is Curry, I am 24 years old and I currently live in New York City."
raw_text_2 = "My name is Adam and im 14 years old!" # No city, successfully sets city=null
# stream_single_turn(
#     user_msg=f"Extract this schema: {schema}\n\nFrom: {raw_text_2}",
#     system="You are a data extractor. Respond ONLY with JSON. No markdown, no explanation",
#     response_type="json"
# ) # Force JSON output

# ---

def analyze_sentiment(user_msg: str) -> dict:
    res = client.models.generate_content_stream(
        model=model, config=types.GenerateContentConfig(
                system_instruction="""
                You are a sentiment analysis engine. Respond only with JSON, no markdown.
                Format this exactly:
                {"sentiment": "positive" or "negative" or "neutral",
                "confidence": float between 0.0 and 1.0,
                "emotions": ["array of detected emotion strings"],
                "summary": "one sentence explanation"}
                """,
                max_output_tokens=540,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=240),
        ),
        contents=user_msg
    )
    full_text = ""
    for chunk in res:
        if chunk.text:
            full_text += chunk.text
    return json.loads(full_text.strip())

# print(analyze_sentiment("I'm not having a good day today for some reason. But I did get some homework done."))

# ---

def summarize_extraction(text: str): # 2 step pipeline
    summary = stream_single_turn(
        user_msg=f"Summarize this within 2-3 sentences: {text}",
        system="You are a concise summarizer"
    )

    extraction = stream_single_turn(
        user_msg=f"Extract this schema: {schema}\n\nFrom: {summary}",
        system="You are a data extractor. Respond ONLY with JSON. No markdown, no explanation",
        response_type="json"
    )

    return {
        "summary": summary,
        "extraction": extraction
    }

# r = summarize_extraction(
    """
    Marcus Thompson is a 34-year-old data scientist living in Chicago.
    He graduated from MIT with a degree in computer science and has been working
    at a healthcare startup for the past 6 years. In his spare time, Marcus
    enjoys playing jazz piano and coaching youth basketball on weekends.
    """
# )
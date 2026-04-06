from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a short story about a cat who can talk.",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant that writes short stories.",
        max_output_tokens=600,
        temperature=0.5,
        top_p=0.95,
        top_k=40,
        frequency_penalty=0,
        presence_penalty=0,
        response_mime_type="text/plain",
    ),
)

print("Ran")
print(response.text)

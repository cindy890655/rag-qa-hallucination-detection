"""
Quick test: check the Gemini API key works and the model responds.
"""
import os
from google import genai

# read key from environment variable (safer than hardcoding)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY first:  export GEMINI_API_KEY='your_key_here'")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly one word: OK",
)
print("Model replied:", response.text.strip())
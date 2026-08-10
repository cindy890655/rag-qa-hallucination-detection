import os
from openai import OpenAI

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("Set GROQ_API_KEY first")

client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
)
print("Model replied:", resp.choices[0].message.content.strip())

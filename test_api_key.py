import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No API Key found.")
    exit(1)

genai.configure(api_key=api_key)

print("🔍 Fetching available models for your API key...")
try:
    models = genai.list_models()
    available_names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    print(f"\n✅ SUCCESS! Your key supports these models:")
    for name in available_names:
        print(f" - {name}")
    
    if available_names:
        print(f"\n👉 Update your .env MODEL_NAME to one of these (e.g., '{available_names[0]}')")
except Exception as e:
    print(f"❌ FAILED to list models: {e}")

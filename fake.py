import google.generativeai as genai
import os

# Replace with your actual key or set environment variable
genai.configure(api_key="API_KEY")

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
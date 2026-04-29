import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# Initialize clients for different models
openai = OpenAI()
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
groq_url = "https://api.groq.com/openai/v1"
gemma_url = "http://localhost:11434/v1"
ollama_url = "http://localhost:11434/v1"

gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=gemini_url)
groq = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=groq_url)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)
gemma = OpenAI(api_key="ollama", base_url=gemma_url)
cohere = OpenAI(base_url="https://api.cohere.ai/compatibility/v1", api_key=os.getenv("COHERE_API_KEY"))

# System message for the assistant
system_message = """
You are a helpful assistant for an Airline called PIA-AI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return response.choices[0].message.content

# Create Gradio interface
if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()

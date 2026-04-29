import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv('COHERE_API_KEY')

if not api_key:
    print("No API key was found - please check your .env file")
elif api_key.strip() != api_key:
    print("API key has extra whitespace - please fix")
else:
    print("API key found and looks good!")

# Set up OpenAI client for Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Get website URL from user
url = input("Enter URL to summarize: ")
website_content = fetch_website_contents(url)

# Create messages for the LLM
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Summarize this website:\n\n{website_content}"}
]

# Get response from LLM
response = client.chat.completions.create(
    model="gemma3:270m",
    messages=messages
)

print("\n=== Summary ===")
print(response.choices[0].message.content)

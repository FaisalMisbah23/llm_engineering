import os
import json
import random
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(override=True)

# Pydantic model for job posting validation
class JobPosting(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    company: str = Field(min_length=2, max_length=120)
    location: str = Field(min_length=2, max_length=80)
    seniority: str
    salary_min: int = Field(ge=20000, le=500000)
    salary_max: int = Field(ge=25000, le=700000)
    skills: List[str]
    description: str = Field(min_length=40, max_length=1200)

    @classmethod
    def normalize(cls, row: Dict) -> Dict:
        row["seniority"] = row.get("seniority", "").lower().strip()
        row["skills"] = [skill.strip() for skill in row.get("skills", []) if str(skill).strip()]
        if row["salary_min"] > row["salary_max"]:
            row["salary_min"], row["salary_max"] = row["salary_max"], row["salary_min"]
        return row

# Create OpenAI client for Groq
def make_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    base_url = "https://api.groq.com/openai/v1"
    return OpenAI(api_key=api_key, base_url=base_url)

SYSTEM_PROMPT = """
You are an expert HR professional writing detailed, realistic job postings.
The user will give you a job title, location, and seniority level.
Write a comprehensive job description that sounds extremely realistic.
Include realistic requirements and responsibilities.
"""

# Function to generate job posting
def generate_job_posting(job_title: str, location: str, seniority: str) -> Dict:
    client = make_client()
    user_prompt = f"Create a job posting for: {job_title} in {location} at {seniority} level."
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    description = response.choices[0].message.content
    
    return {
        "title": job_title,
        "location": location,
        "seniority": seniority,
        "description": description,
        "company": "TechCorp Inc.",
        "salary_min": random.randint(60000, 120000),
        "salary_max": random.randint(120001, 200000),
        "skills": ["Python", "Communication", "Problem Solving", "Teamwork"]
    }

if __name__ == "__main__":
    result = generate_job_posting("Software Engineer", "San Francisco", "Mid-level")
    print(json.dumps(result, indent=2))

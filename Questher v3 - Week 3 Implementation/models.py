import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


EXPERTISE_PROMPTS: Dict[str, str] = {
    "Software Development": "You are a senior software engineer. Be precise, show code when useful, and explain tradeoffs.",
    "DevOps": "You are a DevOps/SRE expert. Focus on reliability, observability, CI/CD, and practical runbooks.",
    "Data Science": "You are a data scientist. Provide correct statistical reasoning, clear steps, and reproducible code.",
    "Machine Learning": "You are an ML engineer. Be practical: data, eval, deployment, and failure modes.",
    "Artificial Intelligence": "You are an AI engineer. Explain concepts clearly and give implementation guidance.",
    "Cybersecurity": "You are a security engineer. Prefer safe defaults, threat modeling, and actionable mitigations.",
    "General": "You are a helpful technical assistant. Be accurate and concise.",
}


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str


class ModelManager:

    def __init__(self) -> None:
        self.provider_configs: Dict[str, ProviderConfig] = {
            "GROQ": ProviderConfig(
                base_url="https://api.groq.com/openai/v1",
                api_key_env="GROQ_API_KEY",
            ),
            "GEMINI": ProviderConfig(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="GOOGLE_API_KEY",
            ),
            "COHERE": ProviderConfig(
                base_url="https://api.cohere.ai/compatibility/v1",
                api_key_env="COHERE_API_KEY",
            ),
            "OLLAMA": ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key_env="ollama",
            ),
        }

        self.providers: Dict[str, List[str]] = {
            "GROQ": [
                "llama-3.1-8b-instant",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
            ],
            "GEMINI": [
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ],
            "COHERE": [
                "command-a-03-2025",
            ],
            "OLLAMA": [
                "llama3:8b",
                "llama3.2:1b",
                "gemma3:270m",
            ],
        }

        self.stt_model_default = ""

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    def get_models(self, provider: str) -> List[str]:
        return self.providers.get(provider.upper(), [])

    def list_expertise(self) -> List[str]:
        return list(EXPERTISE_PROMPTS.keys())

    def system_prompt(self, expertise: str) -> str:
        return EXPERTISE_PROMPTS.get(expertise, EXPERTISE_PROMPTS["General"])

    def get_client(self, provider: str) -> OpenAI:
        p = provider.upper()
        if p not in self.provider_configs:
            raise ValueError(f"Unknown provider: {provider}")

        cfg = self.provider_configs[p]
        api_key = os.getenv(cfg.api_key_env) or cfg.api_key_fallback
        if not api_key:
            raise ValueError(f"Missing {cfg.api_key_env}")

        return OpenAI(api_key=api_key, base_url=cfg.base_url)
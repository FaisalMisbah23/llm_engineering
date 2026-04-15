import tempfile
from typing import Literal

from openai import OpenAI


def transcribe_audio(
    client: OpenAI,
    audio_path: str,
    model: str = "whisper-large-v3-turbo",
) -> str:

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f)
    return resp.text
"""
Voice/text abstraction for multilingual and code-mixed inputs.
"""

import re
from typing import Dict


LANGUAGE_HINTS = {
    "hi": ["nahi", "hai", "mera", "kyun", "abhi", "chahiye"],
    "ta": ["illa", "enna", "inga", "unga"],
    "te": ["ledu", "enti", "meeru", "ippudu"],
    "bn": ["na", "kintu", "ami", "keno"],
}


def detect_language(text: str) -> str:
    lower = text.lower()
    for code, hints in LANGUAGE_HINTS.items():
        if any(token in lower for token in hints):
            return code
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "unknown"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def asr_transcribe(audio_blob: str) -> str:
    """
    Mock ASR function.
    Accepts pseudo audio payload and returns plain text.
    """
    return normalize_text(audio_blob.replace("[AUDIO]", "").strip())


def tts_synthesize(text: str, language: str = "en") -> str:
    """
    Mock TTS function.
    Returns an encoded marker for audio generation.
    """
    return f"[TTS:{language}] {normalize_text(text)}"


def ingest_customer_input(payload: str, channel: str = "text") -> Dict[str, str]:
    if channel == "voice":
        text = asr_transcribe(payload)
    else:
        text = normalize_text(payload)
    language = detect_language(text)
    return {"normalized_text": text, "language": language, "channel": channel}


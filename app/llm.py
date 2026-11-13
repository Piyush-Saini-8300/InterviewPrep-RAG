"""Thin LLM wrappers. Everything degrades gracefully when no provider is configured."""
from __future__ import annotations

import json
import re

from .config import Settings


class LLM:
    provider = "none"
    available = False

    def generate(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        raise RuntimeError("No LLM provider configured")


class AnthropicLLM(LLM):
    provider = "anthropic"
    available = True

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system, user, max_tokens=1024, temperature=0.3):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()


class OpenAILLM(LLM):
    provider = "openai"
    available = True

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system, user, max_tokens=1024, temperature=0.3):
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()


def make_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    try:
        if provider == "anthropic" or (provider == "auto" and settings.anthropic_api_key):
            return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
        if provider == "openai" or (provider == "auto" and settings.openai_api_key):
            return OpenAILLM(settings.openai_api_key, settings.openai_model)
    except Exception as exc:
        print(f"[llm] could not initialise provider '{provider}': {exc}")
    return LLM()


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates ``` fences and chatter)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

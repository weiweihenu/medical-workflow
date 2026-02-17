from __future__ import annotations

import json
import re
from typing import Any, Dict, Generator

from openai import OpenAI

from app.config import AppConfig


def _pick_config_value(config: AppConfig, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value not in (None, ""):
                return value
    return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        api_key = _pick_config_value(config, "openai_api_key", "api_key", default="")
        base_url = _pick_config_value(config, "openai_base_url", "base_url", default="")
        model = _pick_config_value(config, "model", default="gpt-4o-mini")
        temperature = _pick_config_value(config, "temperature", default=0.2)

        api_key = str(api_key or "").strip()
        base_url = str(base_url or "").strip()

        client_kwargs: Dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = str(model or "gpt-4o-mini")
        self.temperature = _to_float(temperature, 0.2)

    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                text: Any = None

                if isinstance(item, dict):
                    text = item.get("text") or item.get("output_text")
                else:
                    text = getattr(item, "text", None)
                    if text is None:
                        text = getattr(item, "output_text", None)

                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

            return "\n".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _cleanup_json_text(text: str) -> str:
        raw = text.strip()
        if not raw:
            return ""

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if fenced_blocks:
            raw = fenced_blocks[0].strip()

        left = raw.find("{")
        right = raw.rfind("}")
        if left != -1 and right != -1 and right > left:
            raw = raw[left : right + 1]

        return raw.strip()

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1200,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._extract_text(response.choices[0].message.content)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1600,
    ) -> Dict[str, Any]:
        text = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not text:
            return {}

        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass

        cleaned = self._cleanup_json_text(text)
        if not cleaned:
            return {}

        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1200,
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=model or self.model,
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)

            if isinstance(content, str):
                if content:
                    yield content
                continue

            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str) and text:
                    yield text
                continue

            if isinstance(content, list):
                for part in content:
                    text: Any = None
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("output_text")
                    else:
                        text = getattr(part, "text", None)
                        if text is None:
                            text = getattr(part, "output_text", None)

                    if isinstance(text, str) and text:
                        yield text

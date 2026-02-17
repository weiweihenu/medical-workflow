from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """应用配置对象。"""
    openai_api_key: str
    openai_base_url: str | None
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量读取配置。"""
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，请先在环境变量或 .env 中配置。")

        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        model = os.getenv("MODEL", "gpt-4o-mini").strip()

        temp_raw = os.getenv("TEMPERATURE", "0.2").strip()
        try:
            temperature = float(temp_raw)
        except ValueError:
            temperature = 0.2

        return cls(
            openai_api_key=api_key,
            openai_base_url=base_url,
            model=model,
            temperature=temperature,
        )

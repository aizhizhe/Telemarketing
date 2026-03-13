from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


def _parse_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    knowledge_base_dir: Path
    database_path: Path
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    top_k: int = 8
    top_n: int = 4
    clarify_max_rounds: int = 2
    brand_name: str = "北文教育"
    human_handoff: str = "人工顾问将在 30 分钟内回拨，或通过微信继续跟进。"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_values = _parse_env(ENV_PATH)

    def get_value(key: str, default: str) -> str:
        return os.getenv(key) or env_values.get(key, default)

    knowledge_base_dir = Path(
        get_value(
            "TELEMARKETING_KNOWLEDGE_BASE_DIR",
            str(ROOT_DIR / "knowledge_base" / "raw"),
        )
    )
    database_path = Path(
        get_value(
            "TELEMARKETING_DATABASE_PATH",
            str(ROOT_DIR / "runtime" / "telemarketing.db"),
        )
    )
    return Settings(
        knowledge_base_dir=knowledge_base_dir,
        database_path=database_path,
        api_key=get_value("TELEMARKETING_API_KEY", ""),
        base_url=get_value(
            "TELEMARKETING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        chat_model=get_value("TELEMARKETING_CHAT_MODEL", "qwen-plus"),
        embedding_model=get_value("TELEMARKETING_EMBEDDING_MODEL", "text-embedding-v4"),
        embedding_dim=int(get_value("TELEMARKETING_EMBEDDING_DIM", "1024")),
        top_k=int(get_value("TELEMARKETING_TOP_K", "8")),
        top_n=int(get_value("TELEMARKETING_TOP_N", "4")),
        clarify_max_rounds=int(get_value("TELEMARKETING_CLARIFY_MAX_ROUNDS", "2")),
        brand_name=get_value("TELEMARKETING_BRAND_NAME", "北文教育"),
        human_handoff=get_value(
            "TELEMARKETING_HUMAN_HANDOFF",
            "人工顾问将在 30 分钟内回拨，或通过微信继续跟进。",
        ),
    )

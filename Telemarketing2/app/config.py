from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENV_PATH = Path(r"E:/Project/Telemarketing/phonecalljg/.env")


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str
    dashscope_base_url: str
    chat_model: str
    fast_chat_model: str
    project_root: Path
    env_path: Path


def _parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _resolve_env_path() -> Path:
    local_env = PROJECT_ROOT / ".env"
    if local_env.exists():
        return local_env
    return LEGACY_ENV_PATH


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_path = _resolve_env_path()
    env_data = _parse_env(env_path)
    api_key = os.getenv("DASHSCOPE_API_KEY") or env_data.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"DASHSCOPE_API_KEY is missing. Checked {env_path}")
    base_url = (
        os.getenv("DASHSCOPE_BASE_URL")
        or env_data.get("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    chat_model = os.getenv("CHAT_MODEL") or env_data.get("CHAT_MODEL") or "qwen3.5-plus"
    fast_chat_model = os.getenv("FAST_CHAT_MODEL") or env_data.get("FAST_CHAT_MODEL") or "qwen-turbo"
    return Settings(
        dashscope_api_key=api_key,
        dashscope_base_url=base_url,
        chat_model=chat_model,
        fast_chat_model=fast_chat_model,
        project_root=PROJECT_ROOT,
        env_path=env_path,
    )

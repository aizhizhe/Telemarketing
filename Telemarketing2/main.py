from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import get_settings
from app.engine import ConversationEngine
from app.materials import get_material_repository
from app.rules_store import load_rules, save_rules


APP_ROOT = Path(__file__).resolve().parent
INDEX_HTML = APP_ROOT / "static" / "index.html"

app = FastAPI(title="Telemarketing2 Debug Console")


class ChatRequest(BaseModel):
    message: str | None = None
    state: dict[str, Any] | None = None


class RulesUpdateRequest(BaseModel):
    rules: dict[str, str]


@lru_cache(maxsize=1)
def get_engine() -> ConversationEngine:
    return ConversationEngine(materials=get_material_repository())


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ok": True,
        "rules": load_rules(),
        "material_stats": get_material_repository().stats(),
        "env_source": str(settings.env_path),
        "model": settings.chat_model,
        "fast_model": settings.fast_chat_model,
    }


@app.get("/api/rules")
def get_rules() -> dict[str, Any]:
    return {"ok": True, "rules": load_rules()}


@app.put("/api/rules")
def update_rules(payload: RulesUpdateRequest) -> dict[str, Any]:
    saved = save_rules(payload.rules)
    return {"ok": True, "rules": saved}


@app.post("/api/chat/start")
def start_chat(payload: ChatRequest) -> dict[str, Any]:
    return get_engine().start_conversation(payload.state)


@app.post("/api/chat/turn")
def turn_chat(payload: ChatRequest) -> dict[str, Any]:
    return get_engine().process_turn(payload.message or "", payload.state)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=False)

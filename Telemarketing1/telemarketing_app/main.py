from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from telemarketing import TelemarketingEngine


app = FastAPI(title="Telemarketing Workflow Console", version="1.1.0")
engine = TelemarketingEngine()
WEB_ROOT = Path(__file__).resolve().parent / "web"


class ChatRequest(BaseModel):
    external_user_id: str = Field(default="demo-user")
    session_key: str = Field(default="demo-session")
    channel: str = Field(default="phone")
    nickname: str | None = None
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "knowledge_records": engine.knowledge_base.record_count,
        "knowledge_dir": str(engine.settings.knowledge_base_dir),
        "database_path": str(engine.settings.database_path),
    }


@app.get("/system-map")
def system_map() -> dict:
    return engine.describe_system()


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    return engine.chat(
        user_text=request.message,
        external_user_id=request.external_user_id,
        session_key=request.session_key,
        channel=request.channel,
        nickname=request.nickname,
    )

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


TRACE_ROOT = PROJECT_ROOT / "logs" / "conversations"


class ConversationTraceLogger:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or TRACE_ROOT

    def log_event(
        self,
        *,
        session_id: str,
        event_type: str,
        user_message: str | None,
        assistant_reply: str,
        state: dict[str, Any],
        debug: dict[str, Any],
    ) -> dict[str, str]:
        now = datetime.now()
        day_dir = self.root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        log_path = day_dir / f"{session_id}.jsonl"
        payload = {
            "timestamp": now.isoformat(timespec="seconds"),
            "session_id": session_id,
            "event_type": event_type,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "state": state,
            "debug": debug,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {
            "session_id": session_id,
            "log_path": str(log_path),
        }

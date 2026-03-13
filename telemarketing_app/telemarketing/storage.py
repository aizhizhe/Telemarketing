from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .knowledge_base import KnowledgeBase
from .models import ConversationState


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    nickname TEXT,
    phone TEXT,
    wechat_id TEXT,
    tags_json TEXT DEFAULT '{}',
    status TEXT DEFAULT 'new',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(channel, external_user_id)
);
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    channel TEXT NOT NULL,
    session_key TEXT NOT NULL UNIQUE,
    current_intent TEXT DEFAULT 'need_clarify',
    clarify_round INTEGER DEFAULT 0,
    summary_text TEXT DEFAULT '',
    risk_flag TEXT DEFAULT 'none',
    handoff_status TEXT DEFAULT 'none',
    state_json TEXT DEFAULT '{}',
    known_slots_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    message_type TEXT DEFAULT 'text',
    content TEXT NOT NULL,
    structured_json TEXT,
    intent TEXT,
    model_name TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    lead_source TEXT NOT NULL,
    name TEXT,
    company TEXT,
    contact_phone TEXT,
    contact_wechat TEXT,
    demand_summary TEXT,
    lead_level TEXT DEFAULT 'warm',
    status TEXT DEFAULT 'new',
    owner_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leads_user_id ON leads(user_id);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    ticket_no TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    order_no TEXT,
    issue_desc TEXT NOT NULL,
    evidence_json TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'open',
    assignee_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    content_text TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    version_no INTEGER DEFAULT 1,
    source_file TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id ON knowledge_chunks(document_id);

CREATE TABLE IF NOT EXISTS conversation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_events_conversation_id ON conversation_events(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_events_event_type ON conversation_events(event_type);
"""


class TelemarketingStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as conn:
            conn.executescript(SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def sync_knowledge_base(self, knowledge_base: KnowledgeBase) -> None:
        with self._session() as conn:
            count = conn.execute("SELECT COUNT(*) AS count FROM knowledge_documents").fetchone()["count"]
            if count:
                return
            now = self._now()
            for document in knowledge_base.knowledge_documents():
                cursor = conn.execute(
                    """
                    INSERT INTO knowledge_documents (
                        doc_title, doc_type, content_text, status, version_no, source_file, created_at, updated_at
                    ) VALUES (?, ?, ?, 'active', 1, ?, ?, ?)
                    """,
                    (
                        document["doc_title"],
                        document["doc_type"],
                        document["content_text"],
                        document["source_file"],
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        document_id, chunk_text, chunk_index, embedding_id, created_at
                    ) VALUES (?, ?, 0, NULL, ?)
                    """,
                    (cursor.lastrowid, document["content_text"], now),
                )

    def get_or_create_conversation(
        self,
        *,
        external_user_id: str,
        session_key: str,
        channel: str,
        nickname: str | None = None,
    ) -> ConversationState:
        now = self._now()
        with self._session() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE channel = ? AND external_user_id = ?",
                (channel, external_user_id),
            ).fetchone()
            if user is None:
                cursor = conn.execute(
                    """
                    INSERT INTO users (
                        external_user_id, channel, nickname, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (external_user_id, channel, nickname, now, now),
                )
                user_id = cursor.lastrowid
            else:
                user_id = user["id"]
                if nickname and nickname != user["nickname"]:
                    conn.execute(
                        "UPDATE users SET nickname = ?, updated_at = ? WHERE id = ?",
                        (nickname, now, user_id),
                    )

            row = conn.execute(
                "SELECT * FROM conversations WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    """
                    INSERT INTO conversations (
                        user_id, channel, session_key, current_intent, clarify_round, summary_text,
                        risk_flag, handoff_status, state_json, known_slots_json, created_at, updated_at
                    ) VALUES (?, ?, ?, 'need_clarify', 0, '', 'none', 'none', '{}', '{}', ?, ?)
                    """,
                    (user_id, channel, session_key, now, now),
                )
                return ConversationState(conversation_id=cursor.lastrowid, user_id=user_id)

            state_payload = json.loads(row["state_json"] or "{}")
            return ConversationState(
                conversation_id=row["id"],
                user_id=row["user_id"],
                current_intent=state_payload.get("current_intent", row["current_intent"]),
                clarify_round=state_payload.get("clarify_round", row["clarify_round"]),
                risk_flag=state_payload.get("risk_flag", row["risk_flag"]),
                handoff_status=state_payload.get("handoff_status", row["handoff_status"]),
                sales_stage=state_payload.get("sales_stage", "opening"),
                known_slots=json.loads(row["known_slots_json"] or "{}"),
                lead_id=state_payload.get("lead_id"),
                ticket_id=state_payload.get("ticket_id"),
                summary_text=row["summary_text"] or "",
                last_reply_type=state_payload.get("last_reply_type", "answer"),
            )

    def save_state(self, state: ConversationState) -> None:
        if state.conversation_id is None:
            raise ValueError("conversation_id is required")
        with self._session() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET current_intent = ?, clarify_round = ?, summary_text = ?, risk_flag = ?,
                    handoff_status = ?, state_json = ?, known_slots_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    state.current_intent,
                    state.clarify_round,
                    state.summary_text,
                    state.risk_flag,
                    state.handoff_status,
                    json.dumps(state.to_dict(), ensure_ascii=False),
                    json.dumps(state.known_slots, ensure_ascii=False),
                    self._now(),
                    state.conversation_id,
                ),
            )

    def append_message(
        self,
        *,
        conversation_id: int,
        role: str,
        content: str,
        intent: str | None = None,
        structured: dict | None = None,
        model_name: str | None = None,
    ) -> None:
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, message_type, content, structured_json, intent, model_name, created_at
                ) VALUES (?, ?, 'text', ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(structured, ensure_ascii=False) if structured else None,
                    intent,
                    model_name,
                    self._now(),
                ),
            )

    def create_event(self, *, conversation_id: int, event_type: str, payload: dict) -> None:
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO conversation_events (conversation_id, event_type, event_payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, event_type, json.dumps(payload, ensure_ascii=False), self._now()),
            )

    def create_lead(self, state: ConversationState, lead_level: str = "warm") -> int:
        now = self._now()
        slots = state.known_slots
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO leads (
                    user_id, conversation_id, lead_source, name, company, contact_phone, contact_wechat,
                    demand_summary, lead_level, status, created_at, updated_at
                ) VALUES (?, ?, 'phone', ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                """,
                (
                    state.user_id,
                    state.conversation_id,
                    slots.get("name"),
                    slots.get("company"),
                    slots.get("contact_phone"),
                    slots.get("contact_wechat"),
                    slots.get("demand_summary"),
                    lead_level,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def create_ticket(self, state: ConversationState, category: str = "complaint") -> tuple[int, str]:
        ticket_no = f"TK{datetime.now(UTC):%Y%m%d%H%M%S}"
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tickets (
                    user_id, conversation_id, ticket_no, category, order_no, issue_desc,
                    evidence_json, priority, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'medium', 'open', ?, ?)
                """,
                (
                    state.user_id,
                    state.conversation_id,
                    ticket_no,
                    category,
                    state.known_slots.get("order_no"),
                    state.known_slots.get("issue_desc") or state.known_slots.get("demand_summary") or "",
                    json.dumps(
                        {
                            "contact_phone": state.known_slots.get("contact_phone"),
                            "contact_wechat": state.known_slots.get("contact_wechat"),
                            "evidence_note": state.known_slots.get("evidence_note"),
                        },
                        ensure_ascii=False,
                    ),
                    self._now(),
                    self._now(),
                ),
            )
            return int(cursor.lastrowid), ticket_no

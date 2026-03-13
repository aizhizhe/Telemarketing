from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class KnowledgeRecord:
    kb_type: str
    title: str
    question: str
    answer: str
    scene: str
    keywords: tuple[str, ...]
    source_file: str
    text: str


@dataclass(frozen=True)
class PainPoint:
    grade: str
    subject: str
    content: str


@dataclass(frozen=True)
class RetrievalHit:
    kb_type: str
    title: str
    question: str
    answer: str
    scene: str
    keywords: tuple[str, ...]
    source_file: str
    text: str
    score: float
    matched_variant: str = ""
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationState:
    conversation_id: int | None = None
    user_id: int | None = None
    current_intent: str = "need_clarify"
    clarify_round: int = 0
    risk_flag: str = "none"
    handoff_status: str = "none"
    sales_stage: str = "opening"
    known_slots: dict[str, str] = field(default_factory=dict)
    lead_id: int | None = None
    ticket_id: int | None = None
    summary_text: str = ""
    last_reply_type: str = "answer"

    def merge_slots(self, updates: dict[str, str]) -> None:
        for key, value in updates.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            existing = str(self.known_slots.get(key, "")).strip()
            if key == "demand_summary" and existing and text not in existing:
                self.known_slots[key] = f"{existing}；{text}"
            else:
                self.known_slots[key] = text

    @property
    def has_contact(self) -> bool:
        return bool(self.known_slots.get("contact_phone") or self.known_slots.get("contact_wechat"))

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "current_intent": self.current_intent,
            "clarify_round": self.clarify_round,
            "risk_flag": self.risk_flag,
            "handoff_status": self.handoff_status,
            "sales_stage": self.sales_stage,
            "known_slots": dict(self.known_slots),
            "lead_id": self.lead_id,
            "ticket_id": self.ticket_id,
            "summary_text": self.summary_text,
            "last_reply_type": self.last_reply_type,
        }

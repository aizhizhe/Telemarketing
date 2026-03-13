from __future__ import annotations

from .knowledge_base import (
    KnowledgeBase,
    extract_grade,
    extract_name,
    extract_order_no,
    extract_phone,
    extract_subjects,
    extract_wechat,
)
from .models import ConversationState, RetrievalHit
from .rag import HybridRAGService
from .settings import Settings, get_settings
from .storage import TelemarketingStorage


RISKY_KEYWORDS = ("包过", "保分", "保证提分", "代考", "伪造", "刷单", "违法", "医疗建议", "法律建议")
COMPLAINT_KEYWORDS = ("投诉", "退费", "退款", "售后", "老师迟到", "老师没来", "课程有问题", "服务有问题")
SALES_KEYWORDS = ("报价", "价格", "费用", "怎么买", "购买", "试听", "试课", "试用", "合作", "演示", "联系销售")
SMALL_TALK_KEYWORDS = ("你好", "在吗", "谢谢", "好的", "行", "嗯", "收到")
OUT_OF_SCOPE_KEYWORDS = ("天气", "股票", "翻译", "写代码", "点外卖")


class TelemarketingEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        knowledge_base: KnowledgeBase | None = None,
        storage: TelemarketingStorage | None = None,
        rag_service: HybridRAGService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.knowledge_base = knowledge_base or KnowledgeBase(self.settings.knowledge_base_dir)
        self.storage = storage or TelemarketingStorage(self.settings.database_path)
        self.storage.sync_knowledge_base(self.knowledge_base)
        self.rag_service = rag_service or HybridRAGService(self.knowledge_base, self.settings)

    def chat(
        self,
        *,
        user_text: str,
        external_user_id: str,
        session_key: str,
        channel: str = "phone",
        nickname: str | None = None,
    ) -> dict:
        state = self.storage.get_or_create_conversation(
            external_user_id=external_user_id,
            session_key=session_key,
            channel=channel,
            nickname=nickname,
        )
        cleaned_text = str(user_text or "").strip()
        if not cleaned_text:
            return {
                "reply": "您直接说下现在最想了解的是课程、校区、试听还是价格，我来帮您快速对上。",
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "collect_question",
                "state": state.to_dict(),
                "retrieved_hits": [],
            }

        self.storage.append_message(
            conversation_id=state.conversation_id,
            role="user",
            content=cleaned_text,
            intent=state.current_intent,
        )

        state.merge_slots(self._extract_slots(cleaned_text))
        intent = self._classify_intent(cleaned_text, state)
        state.current_intent = intent
        state.sales_stage = self._derive_sales_stage(state)

        result = self._dispatch(cleaned_text, state)
        state.last_reply_type = result["reply_type"]
        state.summary_text = self._build_summary(state)

        self.storage.append_message(
            conversation_id=state.conversation_id,
            role="assistant",
            content=result["reply"],
            intent=intent,
            structured={
                "reply_type": result["reply_type"],
                "next_action": result["next_action"],
                "state": state.to_dict(),
            },
            model_name=self.settings.chat_model if self.rag_service.llm_enabled else "rule-based",
        )
        self.storage.save_state(state)
        result["state"] = state.to_dict()
        return result

    def _dispatch(self, user_text: str, state: ConversationState) -> dict:
        if state.current_intent == "risky_request":
            return self._handle_risky(state, user_text)
        if state.current_intent == "complaint_after_sales":
            return self._handle_ticket(state)
        if state.current_intent == "sales_lead":
            return self._handle_lead(state, user_text)
        if state.current_intent == "small_talk":
            return {
                "reply": "在的，您直接说下最想先了解哪一块，我可以先帮您对课程、校区、试听或者价格。",
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "guide_topic",
                "retrieved_hits": [],
            }
        if state.current_intent == "out_of_scope":
            state.handoff_status = "pending"
            return {
                "reply": "这块不属于我这边能确认的内容，我先不乱答。您如果是想了解课程、试听、校区或者顾问对接，我可以马上继续帮您。",
                "intent": state.current_intent,
                "reply_type": "handoff",
                "next_action": "redirect_scope",
                "retrieved_hits": [],
            }
        return self._handle_rag_answer(state, user_text)

    def _classify_intent(self, user_text: str, state: ConversationState) -> str:
        cleaned = user_text.strip()
        if any(keyword in cleaned for keyword in RISKY_KEYWORDS):
            return "risky_request"
        if state.current_intent == "complaint_after_sales" and state.ticket_id is None:
            return "complaint_after_sales"
        if any(keyword in cleaned for keyword in COMPLAINT_KEYWORDS):
            return "complaint_after_sales"
        if state.current_intent == "sales_lead" and state.lead_id is None:
            return "sales_lead"
        if any(keyword in cleaned for keyword in SALES_KEYWORDS):
            return "sales_lead"
        if any(keyword in cleaned for keyword in OUT_OF_SCOPE_KEYWORDS):
            return "out_of_scope"
        if len(cleaned) <= 8 and any(keyword in cleaned for keyword in SMALL_TALK_KEYWORDS):
            return "small_talk"
        if self.knowledge_base.should_use_rag(cleaned):
            return "faq_consult"
        if state.clarify_round < self.settings.clarify_max_rounds:
            return "need_clarify"
        return "faq_consult"

    def _extract_slots(self, user_text: str) -> dict[str, str]:
        slots: dict[str, str] = {}
        grade = extract_grade(user_text)
        if grade:
            slots["grade"] = grade
        subjects = extract_subjects(user_text)
        if subjects:
            slots["subject"] = subjects[0]
        phone = extract_phone(user_text)
        if phone:
            slots["contact_phone"] = phone
        wechat = extract_wechat(user_text)
        if wechat:
            slots["contact_wechat"] = wechat
        name = extract_name(user_text)
        if name:
            slots["name"] = name
        order_no = extract_order_no(user_text)
        if order_no:
            slots["order_no"] = order_no
        if self._extract_schedule_preference(user_text):
            slots["schedule_preference"] = self._extract_schedule_preference(user_text) or ""
        if any(keyword in user_text for keyword in SALES_KEYWORDS + COMPLAINT_KEYWORDS):
            slots["demand_summary"] = user_text
        if any(keyword in user_text for keyword in ("截图", "录音", "聊天记录", "凭证", "证据")):
            slots["evidence_note"] = user_text
        if any(keyword in user_text for keyword in COMPLAINT_KEYWORDS):
            slots["issue_desc"] = user_text
        return slots

    def _derive_sales_stage(self, state: ConversationState) -> str:
        if state.lead_id:
            return "closed"
        if state.has_contact:
            return "invite"
        if state.known_slots.get("grade") and state.known_slots.get("subject"):
            return "diagnose"
        if state.known_slots.get("grade") or state.known_slots.get("subject"):
            return "discovery"
        return "opening"

    def _handle_rag_answer(self, state: ConversationState, user_text: str) -> dict:
        preferred_type = "objection" if any(keyword in user_text for keyword in ("太贵", "太远", "没时间")) else None
        hits = self.rag_service.search_and_rerank(
            user_text,
            top_k=self.settings.top_k,
            top_n=self.settings.top_n,
            preferred_type=preferred_type,
        )
        if hits and hits[0].score >= 4.0:
            reply = self._compose_grounded_reply(state, hits[0])
            state.clarify_round = 0
            self.storage.create_event(
                conversation_id=state.conversation_id,
                event_type="intent_detected",
                payload={"intent": state.current_intent, "hit": hits[0].to_dict()},
            )
            return {
                "reply": reply,
                "intent": state.current_intent,
                "reply_type": "answer",
                "next_action": "continue_sales",
                "retrieved_hits": [hit.to_dict() for hit in hits],
            }
        if state.clarify_round < self.settings.clarify_max_rounds:
            state.clarify_round += 1
            return {
                "reply": self._build_clarify_question(state, user_text),
                "intent": "need_clarify",
                "reply_type": "clarify",
                "next_action": "collect_missing_info",
                "retrieved_hits": [hit.to_dict() for hit in hits],
            }
        state.handoff_status = "pending"
        return {
            "reply": f"这块我先不跟您瞎承诺，避免说偏。您方便留个电话或微信吗？我让人工顾问按您关心的点一对一确认，{self.settings.human_handoff}",
            "intent": "faq_consult",
            "reply_type": "handoff",
            "next_action": "request_contact",
            "retrieved_hits": [hit.to_dict() for hit in hits],
        }

    def _handle_lead(self, state: ConversationState, user_text: str) -> dict:
        missing = self._missing_lead_fields(state)
        if not missing:
            if state.lead_id is None:
                state.lead_id = self.storage.create_lead(state, lead_level=self._infer_lead_level(user_text, state))
                self.storage.create_event(
                    conversation_id=state.conversation_id,
                    event_type="lead_created",
                    payload={"lead_id": state.lead_id, "slots": dict(state.known_slots)},
                )
            state.handoff_status = "pending"
            return {
                "reply": (
                    f"好，我先帮您记下了。孩子现在是{state.known_slots.get('grade', '当前年级')}，"
                    f"重点看{state.known_slots.get('subject', '当前需求')}这块。"
                    f"我们顾问会根据您留的联系方式尽快跟进，{self.settings.human_handoff}"
                ),
                "intent": state.current_intent,
                "reply_type": "collect_lead",
                "next_action": "handoff_to_sales",
                "retrieved_hits": [],
            }
        return {
            "reply": self._build_lead_question(missing[0]),
            "intent": state.current_intent,
            "reply_type": "collect_lead",
            "next_action": f"collect_{missing[0]}",
            "retrieved_hits": [],
        }

    def _handle_ticket(self, state: ConversationState) -> dict:
        missing = self._missing_ticket_fields(state)
        if not missing:
            if state.ticket_id is None:
                state.ticket_id, ticket_no = self.storage.create_ticket(state, category="complaint")
                self.storage.create_event(
                    conversation_id=state.conversation_id,
                    event_type="ticket_created",
                    payload={"ticket_id": state.ticket_id, "ticket_no": ticket_no},
                )
            else:
                ticket_no = f"TK{state.ticket_id}"
            state.handoff_status = "pending"
            return {
                "reply": (
                    f"这边我先帮您受理了，工单号是 {ticket_no}。"
                    "后续会有售后同事根据您提供的信息继续跟进，期间如果还有截图或补充说明，也可以继续发我。"
                ),
                "intent": state.current_intent,
                "reply_type": "handoff",
                "next_action": "after_sales_followup",
                "retrieved_hits": [],
            }
        return {
            "reply": f"理解您现在着急，我先帮您尽快登记。{self._build_ticket_question(missing[0])}",
            "intent": state.current_intent,
            "reply_type": "clarify",
            "next_action": f"collect_{missing[0]}",
            "retrieved_hits": [],
        }

    def _handle_risky(self, state: ConversationState, user_text: str) -> dict:
        state.risk_flag = "high"
        state.handoff_status = "pending"
        self.storage.create_event(
            conversation_id=state.conversation_id,
            event_type="handoff",
            payload={"reason": "risky_request", "message": user_text},
        )
        return {
            "reply": "这类内容我不能替您做承诺或给出不合规建议。如果您需要正式说明，我可以帮您转人工顾问继续处理。",
            "intent": state.current_intent,
            "reply_type": "refuse",
            "next_action": "handoff_to_human",
            "retrieved_hits": [],
        }

    def _compose_grounded_reply(self, state: ConversationState, hit: RetrievalHit) -> str:
        pain_point = self.knowledge_base.get_pain_point(
            state.known_slots.get("grade"),
            state.known_slots.get("subject"),
        )
        opening = hit.answer.strip()
        next_step = self._next_sales_probe(state)
        if self._already_contains_followup(opening, next_step):
            next_step = ""
        if pain_point and state.sales_stage in {"diagnose", "invite"}:
            point_text = pain_point.content.replace("\n", "")
            if len(point_text) > 60:
                point_text = point_text[:60] + "..."
            return f"{opening} 从您刚刚说的情况看，{state.known_slots.get('subject', '这科')}常见卡点也集中在这类问题上。{point_text} {next_step}".strip()
        return f"{opening} {next_step}".strip() if next_step else opening

    def _next_sales_probe(self, state: ConversationState) -> str:
        if not state.known_slots.get("grade"):
            return "顺带问一句，孩子现在几年级了？我好给您对得更准确一点。"
        if not state.known_slots.get("subject"):
            return "您现在最想优先解决哪一科？我按这科给您说得更具体。"
        if not state.has_contact:
            return "如果您愿意，我可以顺手帮您约一节针对性的试听或方案沟通，留个电话或微信就行。"
        if not state.known_slots.get("schedule_preference"):
            return "您看这周还是下周方便，我们可以先把试听时间帮您预留一下。"
        return ""

    def _build_clarify_question(self, state: ConversationState, user_text: str) -> str:
        if not state.known_slots.get("grade"):
            return "我先确认一个关键信息，孩子现在几年级了？"
        if not state.known_slots.get("subject"):
            return "目前更想先解决哪一科，语文、数学还是英语这类？"
        if "校区" in user_text or "远" in user_text:
            return "您更想先了解线下校区，还是也可以接受线上一对一？"
        if "价格" in user_text or "报价" in user_text:
            return "您想先看哪种形式，线下一对一、线上一对一，还是先试听后再定？"
        return "您现在最想先解决的是课程效果、试听安排，还是校区/上课方式？"

    def _build_lead_question(self, field_name: str) -> str:
        if field_name == "grade":
            return "可以，我先帮您对上需求。孩子现在几年级了？"
        if field_name == "subject":
            return "明白，那目前最想优先提升哪一科？"
        if field_name == "contact":
            return "方便留个电话或微信吗？我这边好把试听或方案安排发给您。"
        if field_name == "demand_summary":
            return "您这次最想先解决的，是提分、学习习惯，还是某一科的明显薄弱点？"
        return "您再补充一下最关键的信息，我这边给您继续往下对。"

    def _build_ticket_question(self, field_name: str) -> str:
        if field_name == "issue_desc":
            return "您先简单说下具体问题是什么，我按问题先建档。"
        if field_name == "contact":
            return "方便留个电话或微信吗？售后处理时好及时联系您。"
        if field_name == "order_no":
            return "如果方便的话，把订单号或上课记录发我一下，处理会更快。"
        return "您再补一条关键信息，我先帮您登记进去。"

    def _missing_lead_fields(self, state: ConversationState) -> list[str]:
        missing: list[str] = []
        if not state.known_slots.get("grade"):
            missing.append("grade")
        if not state.known_slots.get("subject"):
            missing.append("subject")
        if not state.has_contact:
            missing.append("contact")
        if not state.known_slots.get("demand_summary"):
            missing.append("demand_summary")
        return missing

    def _missing_ticket_fields(self, state: ConversationState) -> list[str]:
        missing: list[str] = []
        if not state.known_slots.get("issue_desc"):
            missing.append("issue_desc")
        if not state.has_contact:
            missing.append("contact")
        if not state.known_slots.get("order_no"):
            missing.append("order_no")
        return missing

    def _infer_lead_level(self, user_text: str, state: ConversationState) -> str:
        if any(keyword in user_text for keyword in ("报价", "价格", "试听", "试课", "联系销售")):
            return "hot"
        if state.has_contact:
            return "warm"
        return "cold"

    def _build_summary(self, state: ConversationState) -> str:
        slots = []
        for key in ("grade", "subject", "contact_phone", "contact_wechat", "schedule_preference"):
            value = state.known_slots.get(key)
            if value:
                slots.append(f"{key}={value}")
        return (
            f"intent={state.current_intent}; stage={state.sales_stage}; "
            f"clarify_round={state.clarify_round}; {'; '.join(slots)}"
        )

    @staticmethod
    def _extract_schedule_preference(text: str) -> str | None:
        for marker in ("今天", "明天", "后天", "周末", "下周", "晚上", "白天"):
            if marker in text:
                return marker
        return None

    @staticmethod
    def _already_contains_followup(opening: str, next_step: str) -> bool:
        if not next_step:
            return True
        checks = {
            "grade": ("几年级", "哪个年级"),
            "subject": ("哪一科", "哪科"),
            "contact": ("电话", "微信"),
            "schedule": ("这周", "下周", "试听时间"),
        }
        if "几年级" in next_step:
            return any(token in opening for token in checks["grade"])
        if "哪一科" in next_step:
            return any(token in opening for token in checks["subject"])
        if "电话或微信" in next_step:
            return any(token in opening for token in checks["contact"])
        if "这周还是下周" in next_step:
            return any(token in opening for token in checks["schedule"])
        return False

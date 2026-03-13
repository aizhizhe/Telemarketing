from __future__ import annotations

import re
from itertools import cycle

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
from .playbook import (
    DISCOVERY_QUESTIONS,
    HUMAN_PREFIXES,
    OBJECTION_GUIDES,
    REPHRASE_PREFIXES,
    SMALL_TALK_REDIRECTS,
    SUPPLEMENTAL_FAQ,
    TOPIC_KEYWORDS,
)
from .rag import HybridRAGService
from .settings import Settings, get_settings
from .storage import TelemarketingStorage


RISKY_KEYWORDS = ("包过", "保分", "保证提分", "代考", "伪造", "刷单", "违法", "医疗建议", "法律建议")
COMPLAINT_KEYWORDS = ("投诉", "退费", "退款", "售后", "老师迟到", "老师没来", "课程有问题", "服务有问题")
SMALL_TALK_KEYWORDS = ("你好", "在吗", "你好呀", "喂", "有人吗", "谢谢", "收到")
OUT_OF_SCOPE_KEYWORDS = ("天气", "股票", "翻译", "写代码", "点外卖", "电影", "旅游")
SALES_KEYWORDS = ("报价", "价格", "费用", "多少钱", "怎么买", "试听", "试课", "合作", "演示", "联系销售", "报名")
OBJECTION_PATTERNS = {
    "busy": ("没时间", "很忙", "开会", "不方便", "稍后再说"),
    "expensive": ("太贵", "贵", "预算", "便宜点", "优惠"),
    "far": ("太远", "远", "不方便去", "跑不过去"),
    "no_need": ("不需要", "不用了", "先不用", "不考虑", "没兴趣"),
    "child_unwilling": ("孩子不愿意", "孩子不想学", "孩子不配合"),
    "already_have": ("已经补课", "在补课", "已经报班", "已经有老师"),
}
CONTACT_REFUSAL = ("不方便留电话", "不想留电话", "不方便留微信", "不留电话", "不留微信")
POSITIVE_SIGNALS = ("可以", "行", "好", "安排", "约", "留", "加微信", "发我", "那就")
END_MARKERS = ("先这样", "再见", "挂了", "不聊了")


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
        self._redirect_cycle = cycle(SMALL_TALK_REDIRECTS)

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
        cleaned = self._clean_text(user_text)
        if not cleaned:
            reply = "您直接说下最想了解的是校区、课程、试听还是价格，我这边按重点给您说。"
            return {
                "reply": reply,
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "collect_question",
                "state": state.to_dict(),
                "retrieved_hits": [],
            }

        state.turn_index += 1
        state.user_history.append(cleaned)
        state.user_history = state.user_history[-12:]

        self.storage.append_message(
            conversation_id=state.conversation_id,
            role="user",
            content=cleaned,
            intent=state.current_intent,
        )

        extracted = self._extract_slots(cleaned)
        state.merge_slots(extracted)
        if any(item in cleaned for item in CONTACT_REFUSAL):
            state.contact_refused = True

        analysis = self._analyze_turn(cleaned, state)
        state.current_intent = analysis["intent"]
        if analysis["topic"]:
            state.last_topic = analysis["topic"]

        result = self._build_response(cleaned, state, analysis)
        reply = self._finalize_reply(result["reply"], state)
        result["reply"] = reply

        state.reply_history.append(reply)
        state.reply_history = state.reply_history[-12:]
        state.last_reply_type = result["reply_type"]
        state.sales_stage = self._derive_sales_stage(state, analysis, result["next_action"])
        state.summary_text = self._build_summary(state)

        self.storage.append_message(
            conversation_id=state.conversation_id,
            role="assistant",
            content=reply,
            intent=state.current_intent,
            structured={
                "reply_type": result["reply_type"],
                "next_action": result["next_action"],
                "analysis": analysis,
                "state": state.to_dict(),
            },
            model_name=self.settings.chat_model if self.rag_service.llm_enabled else "rule-based",
        )
        self.storage.save_state(state)
        result["state"] = state.to_dict()
        return result

    def _analyze_turn(self, user_text: str, state: ConversationState) -> dict:
        objection_type = self._match_objection(user_text)
        topic = self._detect_topic(user_text)
        repeat_count = sum(1 for item in state.user_history[:-1] if item == user_text)
        angry = any(token in user_text for token in ("烦", "骚扰", "骗子", "滚", "投诉你", "别打了"))
        positive = any(token in user_text for token in POSITIVE_SIGNALS)
        if any(token in user_text for token in RISKY_KEYWORDS):
            intent = "risky_request"
        elif any(token in user_text for token in COMPLAINT_KEYWORDS):
            intent = "complaint_after_sales"
        elif state.current_intent == "complaint_after_sales" and state.ticket_id is None:
            intent = "complaint_after_sales"
        elif self._is_out_of_scope(user_text):
            intent = "out_of_scope"
        elif len(user_text) <= 6 and any(token in user_text for token in SMALL_TALK_KEYWORDS):
            intent = "small_talk"
        elif any(token in user_text for token in SALES_KEYWORDS) or state.has_contact or positive:
            intent = "sales_lead"
        elif self.knowledge_base.should_use_rag(user_text) or objection_type:
            intent = "faq_consult"
        else:
            intent = "need_clarify"
        return {
            "intent": intent,
            "topic": topic,
            "objection_type": objection_type,
            "repeat_count": repeat_count,
            "angry": angry,
            "positive": positive,
        }

    def _build_response(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        if analysis["intent"] == "risky_request":
            return self._handle_risky(state, user_text)
        if analysis["intent"] == "complaint_after_sales":
            return self._handle_complaint(state)
        if analysis["intent"] == "out_of_scope":
            return self._handle_out_of_scope()
        if analysis["intent"] == "small_talk":
            return self._handle_small_talk()
        if analysis["objection_type"]:
            return self._handle_objection(user_text, state, analysis)
        if analysis["intent"] == "sales_lead":
            return self._handle_sales(user_text, state, analysis)
        return self._handle_faq(user_text, state, analysis)

    def _handle_sales(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
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
            state.conversation_outcome = "lead_created"
            follow = ""
            if not state.known_slots.get("schedule_preference"):
                follow = "您如果方便，也可以顺手告诉我这周还是下周更好约。"
            reply = (
                f"{self._pick_prefix('warm', state)}我先帮您记下了，孩子现在是"
                f"{state.known_slots.get('grade', '当前年级')}，重点看{state.known_slots.get('subject', '当前需求')}。"
                f"顾问会按您留的联系方式尽快跟进，{self.settings.human_handoff}{follow}"
            )
            return {
                "reply": reply,
                "intent": state.current_intent,
                "reply_type": "collect_lead",
                "next_action": "handoff_to_sales",
                "retrieved_hits": [],
            }

        if state.contact_refused and "contact" in missing:
            state.handoff_status = "pending"
            state.conversation_outcome = "soft_close"
            return {
                "reply": "明白，我不强留联系方式。您后面如果想继续了解，可以直接留一句“想约试听”或者“想看方案”，我这边再帮您接上。",
                "intent": state.current_intent,
                "reply_type": "handoff",
                "next_action": "soft_close",
                "retrieved_hits": [],
            }

        next_field = missing[0]
        reply = self._ask_field(next_field, state)
        return {
            "reply": reply,
            "intent": state.current_intent,
            "reply_type": "collect_lead",
            "next_action": f"collect_{next_field}",
            "retrieved_hits": [],
        }

    def _handle_complaint(self, state: ConversationState) -> dict:
        if not state.known_slots.get("issue_desc") and state.known_slots.get("demand_summary"):
            state.known_slots["issue_desc"] = state.known_slots["demand_summary"]

        if not state.known_slots.get("issue_desc"):
            return {
                "reply": f"{self._pick_prefix('complaint', state)}您先简单说下具体问题，我按问题先帮您登记。",
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "collect_issue_desc",
                "retrieved_hits": [],
            }

        if not state.has_contact:
            return {
                "reply": f"{self._pick_prefix('complaint', state)}为了方便售后尽快联系您，麻烦留个电话或微信。",
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "collect_contact",
                "retrieved_hits": [],
            }

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
        state.conversation_outcome = "ticket_created"
        extra = ""
        if not state.known_slots.get("order_no"):
            extra = "如果您方便，后面再补个订单号或截图，处理会更快。"
        return {
            "reply": f"{self._pick_prefix('complaint', state)}这边已经先帮您建单了，工单号是 {ticket_no}。{extra}售后同事会继续跟进。",
            "intent": state.current_intent,
            "reply_type": "handoff",
            "next_action": "after_sales_followup",
            "retrieved_hits": [],
        }

    def _handle_out_of_scope(self) -> dict:
        return {
            "reply": "这块不属于我这边能确认的范围，我先不乱答。您如果是想了解课程、试听、校区或者顾问对接，我可以马上继续帮您。",
            "intent": "out_of_scope",
            "reply_type": "handoff",
            "next_action": "redirect_scope",
            "retrieved_hits": [],
        }

    def _handle_small_talk(self) -> dict:
        return {
            "reply": next(self._redirect_cycle),
            "intent": "small_talk",
            "reply_type": "clarify",
            "next_action": "guide_topic",
            "retrieved_hits": [],
        }

    def _handle_risky(self, state: ConversationState, user_text: str) -> dict:
        state.risk_flag = "high"
        state.handoff_status = "pending"
        state.conversation_outcome = "risk_handoff"
        self.storage.create_event(
            conversation_id=state.conversation_id,
            event_type="handoff",
            payload={"reason": "risky_request", "message": user_text},
        )
        return {
            "reply": "这类内容我不能替您做承诺或给不合规建议。如果您需要正式说明，我可以帮您转人工继续处理。",
            "intent": "risky_request",
            "reply_type": "refuse",
            "next_action": "handoff_to_human",
            "retrieved_hits": [],
        }

    def _handle_objection(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        objection_type = analysis["objection_type"]
        hits = self.rag_service.search_and_rerank(
            user_text,
            top_k=self.settings.top_k,
            top_n=self.settings.top_n,
            preferred_type="objection",
        )
        core_answer = self._build_kb_answer(hits[0], state, repeated=analysis["repeat_count"] > 0) if hits else ""
        if not core_answer:
            core_answer = self._fallback_objection_answer(objection_type)
        bridge = self._next_sales_probe(state, analysis["topic"])
        lead = self._pick_objection_lead(objection_type)
        reply = f"{lead}{core_answer}"
        if bridge:
            reply = f"{reply}{bridge}"
        if objection_type == "no_need" and any(token in user_text for token in END_MARKERS):
            state.conversation_outcome = "closed_by_user"
            reply = "明白，那我就不继续打扰您了。后面如果您想再了解试听或方案，随时找我就行。"
        return {
            "reply": reply,
            "intent": "faq_consult",
            "reply_type": "answer",
            "next_action": "continue_sales",
            "retrieved_hits": [hit.to_dict() for hit in hits],
        }

    def _handle_faq(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        hits = self.rag_service.search_and_rerank(
            user_text,
            top_k=self.settings.top_k,
            top_n=self.settings.top_n,
        )
        if hits and hits[0].score >= 4.0:
            reply = self._compose_faq_reply(state, analysis, hits[0])
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

        supplemental = self._supplemental_answer(analysis["topic"])
        if supplemental:
            return {
                "reply": f"{self._pick_prefix('neutral', state)}{supplemental}{self._next_sales_probe(state, analysis['topic'])}",
                "intent": state.current_intent,
                "reply_type": "answer",
                "next_action": "continue_sales",
                "retrieved_hits": [],
            }

        if state.clarify_round < self.settings.clarify_max_rounds:
            state.clarify_round += 1
            question = self._clarify_question(state, analysis["topic"])
            return {
                "reply": question,
                "intent": "need_clarify",
                "reply_type": "clarify",
                "next_action": "collect_missing_info",
                "retrieved_hits": [],
            }

        state.handoff_status = "pending"
        state.conversation_outcome = "faq_handoff"
        return {
            "reply": f"这块我先不瞎说，免得误导您。您方便留个电话或微信吗？我让顾问按您关心的点一对一给您确认。",
            "intent": "faq_consult",
            "reply_type": "handoff",
            "next_action": "request_contact",
            "retrieved_hits": [],
        }

    def _compose_faq_reply(self, state: ConversationState, analysis: dict, hit: RetrievalHit) -> str:
        lead = self._pick_prefix("neutral", state)
        repeated = analysis["repeat_count"] > 0 or self._is_similar_to_last_topic(state, analysis["topic"])
        if repeated:
            lead = f"{REPHRASE_PREFIXES[analysis['repeat_count'] % len(REPHRASE_PREFIXES)]}"
        core = self._build_kb_answer(hit, state, repeated=repeated)
        pain_point = self.knowledge_base.get_pain_point(
            state.known_slots.get("grade"),
            state.known_slots.get("subject"),
        )
        diagnosis = ""
        if pain_point and state.known_slots.get("subject") and state.sales_stage in {"diagnosis", "proposal", "invite"}:
            short_pain = self._summarize_text(pain_point.content, max_sentences=1, max_chars=46)
            diagnosis = f"按您说的情况，{state.known_slots.get('subject')}常见卡点大多也在这块。{short_pain}"
        probe = self._next_sales_probe(state, analysis["topic"])
        return f"{lead}{core}{diagnosis}{probe}"

    def _build_kb_answer(self, hit: RetrievalHit, state: ConversationState, repeated: bool = False) -> str:
        answer = self._summarize_text(hit.answer, max_sentences=2 if repeated else 3, max_chars=140 if repeated else 180)
        if state.last_topic and state.last_topic == hit.title and repeated:
            return answer
        return answer

    def _fallback_objection_answer(self, objection_type: str) -> str:
        mapping = {
            "busy": "我不占您太久，先把关键点说清楚：如果孩子确实有提升需求，先把年级和科目对上，比现在马上决定报不报更重要。",
            "expensive": "价格这块我不回避，但要先看孩子情况，不然我现在报个数对您也不负责。",
            "far": "如果您不方便跑校区，线上一对一也可以先了解，关键还是匹配孩子当前问题。",
            "no_need": "没关系，我不会硬推。只是如果孩子后面成绩或状态有波动，至少您知道我这边能怎么帮您。",
            "child_unwilling": "孩子不配合确实很常见，所以前面先判断适不适合，再安排试听才更稳。",
            "already_have": "已经有老师也没问题，关键是现在的效果和匹配度是不是到位。",
        }
        return mapping.get(objection_type, "这个顾虑我理解，我先把关键点给您说清楚。")

    def _pick_objection_lead(self, objection_type: str) -> str:
        options = OBJECTION_GUIDES.get(objection_type, HUMAN_PREFIXES["empathy"])
        return options[0]

    def _next_sales_probe(self, state: ConversationState, topic: str | None) -> str:
        if state.conversation_outcome in {"lead_created", "ticket_created"}:
            return ""
        if not state.known_slots.get("grade"):
            return self._ask_field("grade", state)
        if not state.known_slots.get("subject") and topic not in {"cooperation"}:
            return self._ask_field("subject", state)
        if topic == "cooperation" and not state.has_contact:
            return "您先留个电话或微信，我按合作方向让对应同事跟您对接。"
        if topic == "price" and not state.has_contact:
            return "您愿意的话，留个电话或微信，我按孩子情况把更准确的方案和报价范围发给您。"
        if state.known_slots.get("grade") and state.known_slots.get("subject") and not state.has_contact:
            return self._ask_field("contact", state)
        if state.has_contact and not state.known_slots.get("schedule_preference"):
            return self._ask_field("schedule", state)
        return ""

    def _clarify_question(self, state: ConversationState, topic: str | None) -> str:
        if not state.known_slots.get("grade"):
            return "我先确认一个关键信息，孩子现在几年级了？"
        if not state.known_slots.get("subject") and topic != "cooperation":
            return "您现在最想先解决哪一科？我按这一科给您说得更具体。"
        if topic == "price":
            return "您更偏线上还是线下，或者先试听再决定？"
        if topic == "location":
            return "您更想了解线下校区，还是也可以接受线上一对一？"
        return "您现在最想先确认的是课程效果、试听安排，还是上课方式？"

    def _ask_field(self, field_name: str, state: ConversationState) -> str:
        state.asked_fields.append(field_name)
        state.asked_fields = state.asked_fields[-10:]
        options = DISCOVERY_QUESTIONS[field_name]
        turn_seed = sum(1 for item in state.asked_fields if item == field_name)
        question = options[(turn_seed - 1) % len(options)]
        if field_name == "contact" and state.known_slots.get("grade") and state.known_slots.get("subject"):
            return f"目前孩子是{state.known_slots['grade']}，重点看{state.known_slots['subject']}，{question}"
        return question

    def _missing_lead_fields(self, state: ConversationState) -> list[str]:
        missing: list[str] = []
        if not state.known_slots.get("grade"):
            missing.append("grade")
        if not state.known_slots.get("subject") and "合作" not in state.known_slots.get("demand_summary", ""):
            missing.append("subject")
        if not state.known_slots.get("demand_summary") and not (
            state.known_slots.get("grade") and state.known_slots.get("subject")
        ):
            missing.append("demand_summary")
        if not state.has_contact:
            missing.append("contact")
        return missing

    def _detect_topic(self, text: str) -> str | None:
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return topic
        return None

    def _is_out_of_scope(self, text: str) -> bool:
        return any(keyword in text for keyword in OUT_OF_SCOPE_KEYWORDS)

    def _match_objection(self, text: str) -> str | None:
        for objection_type, keywords in OBJECTION_PATTERNS.items():
            if any(keyword in text for keyword in keywords):
                return objection_type
        return None

    def _infer_lead_level(self, user_text: str, state: ConversationState) -> str:
        if any(keyword in user_text for keyword in ("报价", "价格", "试听", "试课", "报名", "联系销售")):
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
            f"intent={state.current_intent}; stage={state.sales_stage}; turns={state.turn_index}; "
            f"outcome={state.conversation_outcome}; {'; '.join(slots)}"
        )

    def _derive_sales_stage(self, state: ConversationState, analysis: dict, next_action: str) -> str:
        if state.current_intent == "complaint_after_sales":
            return "after_sales"
        if state.lead_id:
            return "closing"
        if next_action == "handoff_to_sales":
            return "closing"
        if state.has_contact:
            return "invite"
        if state.known_slots.get("grade") and state.known_slots.get("subject"):
            return "proposal"
        if state.known_slots.get("grade") or state.known_slots.get("subject"):
            return "diagnosis"
        if analysis["topic"] or analysis["intent"] in {"faq_consult", "sales_lead"}:
            return "discovery"
        return "opening"

    def _supplemental_answer(self, topic: str | None) -> str:
        if not topic:
            return ""
        variants = SUPPLEMENTAL_FAQ.get(topic)
        if not variants:
            return ""
        return variants[0]

    def _pick_prefix(self, style: str, state: ConversationState) -> str:
        options = HUMAN_PREFIXES[style]
        index = state.turn_index % len(options)
        return options[index]

    def _is_similar_to_last_topic(self, state: ConversationState, topic: str | None) -> bool:
        return bool(topic and state.last_topic and topic == state.last_topic)

    def _summarize_text(self, text: str, *, max_sentences: int, max_chars: int) -> str:
        cleaned = self._clean_text(text)
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[。！？!?])", cleaned)
        sentences = [part.strip() for part in parts if part.strip()]
        summarized = "".join(sentences[:max_sentences]).strip()
        if not summarized:
            summarized = cleaned
        if len(summarized) > max_chars:
            summarized = summarized[: max_chars - 1].rstrip("，。；; ") + "。"
        return summarized

    def _finalize_reply(self, reply: str, state: ConversationState) -> str:
        cleaned = self._clean_text(reply)
        cleaned = re.sub(r"\s+", "", cleaned)
        if len(cleaned) > 220:
            cleaned = self._summarize_text(cleaned, max_sentences=3, max_chars=220)
        if state.reply_history and cleaned == state.reply_history[-1]:
            cleaned = f"{REPHRASE_PREFIXES[state.turn_index % len(REPHRASE_PREFIXES)]}{cleaned}"
        cleaned = cleaned.replace("？？", "？").replace("。。", "。")
        return cleaned

    @staticmethod
    def _clean_text(text: str) -> str:
        return str(text or "").strip()

    @staticmethod
    def _extract_schedule_preference(text: str) -> str | None:
        for marker in ("今天", "明天", "后天", "周末", "下周", "晚上", "白天"):
            if marker in text:
                return marker
        return None

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
        schedule = self._extract_schedule_preference(user_text)
        if schedule:
            slots["schedule_preference"] = schedule
        if any(keyword in user_text for keyword in SALES_KEYWORDS + COMPLAINT_KEYWORDS):
            slots["demand_summary"] = user_text
        if any(keyword in user_text for keyword in ("截图", "录音", "聊天记录", "凭证", "证据")):
            slots["evidence_note"] = user_text
        if any(keyword in user_text for keyword in COMPLAINT_KEYWORDS):
            slots["issue_desc"] = user_text
        return slots

    @staticmethod
    def _safe_turn_mod(seed: str) -> int:
        return sum(ord(ch) for ch in seed) % 3 + 1

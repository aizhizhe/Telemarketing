from __future__ import annotations

import re
from itertools import cycle
from pathlib import Path

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
    SYSTEM_LAYERS,
    TOPIC_KEYWORDS,
)
from .rag import HybridRAGService
from .settings import Settings, get_settings
from .storage import TelemarketingStorage


RISKY_KEYWORDS = ("包过", "保分", "保证提分", "代考", "违法", "造假", "刷单", "医疗建议", "法律建议")
COMPLAINT_KEYWORDS = ("投诉", "退款", "退费", "售后", "老师迟到", "老师没来", "课程有问题", "服务有问题")
SMALL_TALK_KEYWORDS = ("你好", "在吗", "哈喽", "有人吗", "你好呀", "收到", "谢谢")
OUT_OF_SCOPE_KEYWORDS = ("天气", "股票", "翻译", "写代码", "点外卖", "电影", "旅游")
SALES_KEYWORDS = ("报价", "价格", "费用", "多少钱", "怎么买", "试听", "试课", "体验", "合作", "演示", "报名")
IDENTITY_KEYWORDS = ("你是谁", "你哪位", "你谁啊", "你是干嘛的", "你们是干什么的", "找我什么事", "打电话干嘛")
ROBOT_KEYWORDS = ("你是机器人吗", "你是真人吗", "机器人", "AI吗", "人工吗")
CONTACT_REFUSAL = ("不方便留电话", "不想留电话", "不方便留微信", "不留电话", "不留微信")
POSITIVE_SIGNALS = ("可以", "行", "好", "安排", "约", "留", "加微信", "发我", "那就")
END_MARKERS = ("先这样", "再见", "挂了", "不聊了")
ANGRY_KEYWORDS = ("烦", "骚扰", "骗子", "滚", "投诉你", "别打了")
OBJECTION_PATTERNS = {
    "busy": ("没时间", "很忙", "开会", "不方便", "稍后再说"),
    "expensive": ("太贵", "贵", "预算", "便宜点", "优惠"),
    "far": ("太远", "远", "不方便去", "跑不过去"),
    "no_need": ("不需要", "不用了", "先不用", "不考虑", "没兴趣"),
    "child_unwilling": ("孩子不愿意", "孩子不想学", "孩子不配合"),
    "already_have": ("已经补课", "在补课", "已经报班", "已经有老师"),
}


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
        self._root_dir = Path(__file__).resolve().parent.parent
        self._question_cycle_index = 0

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
            return {
                "reply": "您直接说下最想了解的是校区、课程、试听还是价格，我按重点跟您说。",
                "intent": state.current_intent,
                "reply_type": "clarify",
                "next_action": "collect_question",
                "state": state.to_dict(),
                "retrieved_hits": [],
                "trace": self._build_empty_trace("空消息，进入兜底澄清。"),
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

        extracted = self._extract_slots(cleaned, state)
        state.merge_slots(extracted)
        if any(item in cleaned for item in CONTACT_REFUSAL):
            state.contact_refused = True

        analysis = self._analyze_turn(cleaned, state)
        state.current_intent = analysis["intent"]
        if analysis["topic"]:
            state.last_topic = analysis["topic"]

        result = self._build_response(cleaned, state, analysis)
        reply = self._finalize_reply(result["reply"], state, analysis)
        result["reply"] = reply

        state.reply_history.append(reply)
        state.reply_history = state.reply_history[-12:]
        state.last_reply_type = result["reply_type"]
        state.sales_stage = self._derive_sales_stage(state, analysis, result["next_action"])
        state.summary_text = self._build_summary(state)

        trace = self._build_trace(
            user_text=cleaned,
            state=state,
            extracted=extracted,
            analysis=analysis,
            result=result,
        )
        result["trace"] = trace

        self.storage.append_message(
            conversation_id=state.conversation_id,
            role="assistant",
            content=reply,
            intent=state.current_intent,
            structured={
                "reply_type": result["reply_type"],
                "next_action": result["next_action"],
                "analysis": analysis,
                "trace": trace,
                "state": state.to_dict(),
            },
            model_name=self.settings.chat_model if self.rag_service.llm_enabled else "rule-based",
        )
        self.storage.save_state(state)
        result["state"] = state.to_dict()
        return result

    def describe_system(self) -> dict:
        layers = []
        for item in SYSTEM_LAYERS:
            file_paths = [str((self._root_dir / part.strip()).resolve()) for part in item["file"].split("+")]
            layers.append(
                {
                    **item,
                    "file_paths": file_paths,
                    "file_path": " | ".join(file_paths),
                }
            )
        return {
            "layers": layers,
            "knowledge_files": [
                {"name": path.name, "path": str(path.resolve())}
                for path in sorted(self.settings.knowledge_base_dir.glob("*.xlsx"))
            ],
            "entrypoints": [
                {"name": "聊天页", "path": str((self._root_dir / "web" / "index.html").resolve())},
                {"name": "接口入口", "path": str((self._root_dir / "main.py").resolve())},
                {"name": "对话引擎", "path": str((self._root_dir / "telemarketing" / "engine.py").resolve())},
            ],
        }

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").replace("\u3000", " ")).strip()

    def _extract_slots(self, user_text: str, state: ConversationState) -> dict[str, str]:
        slots: dict[str, str] = {}
        grade = extract_grade(user_text)
        if grade:
            slots["grade"] = grade
        subjects = extract_subjects(user_text)
        if subjects:
            slots["subject"] = "、".join(subjects)
        phone = extract_phone(user_text)
        if phone:
            slots["contact_phone"] = phone
        wechat = extract_wechat(user_text)
        if wechat:
            slots["contact_wechat"] = wechat
        order_no = extract_order_no(user_text)
        if order_no:
            slots["order_no"] = order_no
        name = extract_name(user_text)
        if name:
            slots["name"] = name
        if len(user_text) >= 6:
            if any(token in user_text for token in COMPLAINT_KEYWORDS) and not state.known_slots.get("issue_desc"):
                slots["issue_desc"] = user_text
            elif any(token in user_text for token in SALES_KEYWORDS) or self._detect_topic(user_text) in {
                "price",
                "trial",
                "teacher",
                "online",
                "effect",
                "cooperation",
                "location",
            }:
                slots["demand_summary"] = user_text[:80]
        return slots

    def _analyze_turn(self, user_text: str, state: ConversationState) -> dict:
        objection_type = self._match_objection(user_text)
        topic = self._detect_topic(user_text)
        repeat_count = sum(1 for item in state.user_history[:-1] if self._normalize(item) == self._normalize(user_text))
        angry = any(token in user_text for token in ANGRY_KEYWORDS)
        positive = any(token in user_text for token in POSITIVE_SIGNALS)
        reasons: list[str] = []

        if any(token in user_text for token in RISKY_KEYWORDS):
            intent = "risky_request"
            reasons.append("命中高风险关键词")
        elif any(token in user_text for token in COMPLAINT_KEYWORDS) or (
            state.current_intent == "complaint_after_sales" and state.ticket_id is None
        ):
            intent = "complaint_after_sales"
            reasons.append("命中投诉/售后语义")
        elif any(token in user_text for token in ROBOT_KEYWORDS):
            intent = "identity_check"
            reasons.append("用户在确认是否真人或机器人")
        elif any(token in user_text for token in IDENTITY_KEYWORDS):
            intent = "identity_check"
            reasons.append("用户在确认来电身份或目的")
        elif self._is_out_of_scope(user_text):
            intent = "out_of_scope"
            reasons.append("内容超出业务范围")
        elif len(user_text) <= 6 and any(token in user_text for token in SMALL_TALK_KEYWORDS):
            intent = "small_talk"
            reasons.append("短句寒暄，需要引导进入正题")
        elif objection_type:
            intent = "faq_consult"
            reasons.append(f"命中异议处理：{objection_type}")
        elif any(token in user_text for token in SALES_KEYWORDS) or state.has_contact or positive:
            intent = "sales_lead"
            reasons.append("存在销售推进信号")
        elif self.knowledge_base.should_use_rag(user_text):
            intent = "faq_consult"
            reasons.append("适合走知识问答/RAG")
        else:
            intent = "need_clarify"
            reasons.append("信息不足，进入单轮澄清")

        return {
            "intent": intent,
            "topic": topic,
            "objection_type": objection_type,
            "repeat_count": repeat_count,
            "angry": angry,
            "positive": positive,
            "reasons": reasons,
        }

    def _build_response(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        if analysis["intent"] == "risky_request":
            return self._handle_risky(state, user_text)
        if analysis["intent"] == "complaint_after_sales":
            return self._handle_complaint(state)
        if analysis["intent"] == "identity_check":
            return self._handle_identity(state, user_text)
        if analysis["intent"] == "out_of_scope":
            return self._handle_out_of_scope()
        if analysis["intent"] == "small_talk":
            return self._handle_small_talk()
        if analysis["objection_type"]:
            return self._handle_objection(user_text, state, analysis)
        if analysis["intent"] == "sales_lead":
            return self._handle_sales(user_text, state, analysis)
        return self._handle_faq(user_text, state, analysis)

    def _handle_risky(self, state: ConversationState, user_text: str) -> dict:
        state.risk_flag = "high"
        state.conversation_outcome = "risk_refused"
        reply = (
            f"这类内容我不能帮您处理。"
            f"如果您想了解的是{self.settings.brand_name}的课程、校区、试听或售后，我可以按正规流程继续协助您。"
        )
        return {
            "reply": reply,
            "intent": "risky_request",
            "reply_type": "refuse",
            "next_action": "redirect_to_compliant_scope",
            "retrieved_hits": [],
            "strategy": "拒绝风险请求并收回到合规业务范围",
        }

    def _handle_complaint(self, state: ConversationState) -> dict:
        missing = self._missing_complaint_fields(state)
        if missing:
            state.handoff_status = "collecting"
            prompt_map = {
                "issue_desc": "您简单说下具体是什么问题，我先帮您登记清楚。",
                "order_no": self._pick_question("order_no"),
                "contact": "方便留个联系电话吗？售后老师回访会更快一些。",
            }
            next_key = missing[0]
            return {
                "reply": f"{self._pick_phrase('complaint')} {prompt_map[next_key]}",
                "intent": "complaint_after_sales",
                "reply_type": "collect_lead",
                "next_action": f"collect_{next_key}",
                "retrieved_hits": [],
                "strategy": "投诉处理先补齐工单信息",
            }

        if state.ticket_id is None:
            ticket_id, ticket_no = self.storage.create_ticket(state)
            state.ticket_id = ticket_id
            state.handoff_status = "ticket_created"
            state.conversation_outcome = "ticket_created"
            self.storage.create_event(
                conversation_id=state.conversation_id,
                event_type="ticket_created",
                payload={"ticket_id": ticket_id, "ticket_no": ticket_no},
            )
            reply = (
                f"{self._pick_phrase('complaint')} 已经帮您登记为售后工单，工单号是 {ticket_no}。"
                f" {self.settings.human_handoff}"
            )
        else:
            reply = f"这边已经为您登记过工单，我会继续催进处理进度。{self.settings.human_handoff}"
        return {
            "reply": reply,
            "intent": "complaint_after_sales",
            "reply_type": "handoff",
            "next_action": "ticket_handoff",
            "retrieved_hits": [],
            "strategy": "信息齐全后创建工单并转售后",
        }

    def _handle_identity(self, state: ConversationState, user_text: str) -> dict:
        brand = self.settings.brand_name
        if any(token in user_text for token in ROBOT_KEYWORDS):
            reply = (
                f"我这边是{brand}的智能咨询助手，负责先把家长的问题接住、确认需求和安排后续。"
                "如果您希望真人顾问继续跟进，我也可以帮您转接。"
            )
        else:
            reply = (
                f"我是{brand}这边负责课程咨询的，主要帮家长先了解校区、课程、试听和后续安排。"
                "您不用有压力，我先把您关心的点说清楚。"
            )
        next_question = self._identity_next_question(state)
        return {
            "reply": f"{reply} {next_question}",
            "intent": "identity_check",
            "reply_type": "answer",
            "next_action": "build_trust_then_discover",
            "retrieved_hits": [],
            "strategy": "先建立身份信任，再轻推回业务问题",
        }

    def _handle_out_of_scope(self) -> dict:
        return {
            "reply": "这个不属于我这边的业务范围。我主要能帮您处理课程咨询、试听安排、校区信息和售后问题，您如果是这几类我可以直接接着说。",
            "intent": "out_of_scope",
            "reply_type": "clarify",
            "next_action": "redirect_to_scope",
            "retrieved_hits": [],
            "strategy": "边界说明后拉回主业务",
        }

    def _handle_small_talk(self) -> dict:
        return {
            "reply": next(self._redirect_cycle),
            "intent": "small_talk",
            "reply_type": "clarify",
            "next_action": "redirect_to_business_topic",
            "retrieved_hits": [],
            "strategy": "短寒暄后快速引导进入正题",
        }

    def _handle_objection(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        objection_type = analysis["objection_type"]
        hits = self._retrieve_hits(user_text, preferred_type="objection")
        base = self._pick_objection_phrase(objection_type)
        hit_text = hits[0].answer if hits else self._fallback_objection_answer(objection_type)
        next_step = self._objection_next_step(objection_type, state)
        reply = f"{base} {self._normalize_sentence(hit_text)}"
        if next_step:
            reply = f"{reply} {next_step}"
        return {
            "reply": reply,
            "intent": "faq_consult",
            "reply_type": "answer" if objection_type in {"expensive", "already_have", "child_unwilling"} else "clarify",
            "next_action": "objection_follow_up",
            "retrieved_hits": [hit.to_dict() for hit in hits],
            "strategy": "先接异议，再给事实和下一步",
        }

    def _handle_sales(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        topic = analysis["topic"] or state.last_topic
        hits = self._retrieve_hits(user_text, preferred_type="qa")
        knowledge_text = self._knowledge_reply(hits, topic)
        missing = self._missing_sales_fields(state, topic)

        if state.has_contact and state.lead_id is None:
            state.lead_id = self.storage.create_lead(state)
            state.conversation_outcome = "lead_created"
            self.storage.create_event(
                conversation_id=state.conversation_id,
                event_type="lead_created",
                payload={"lead_id": state.lead_id, "topic": topic or "general"},
            )

        if state.lead_id and not missing:
            state.handoff_status = "lead_ready"
            reply = (
                f"{knowledge_text} 我已经把您的需求记下来了。"
                f" 接下来我给您安排顾问按您提供的情况继续跟进，{self.settings.human_handoff}"
            ).strip()
            return {
                "reply": reply,
                "intent": "sales_lead",
                "reply_type": "collect_lead",
                "next_action": "consultant_follow_up",
                "retrieved_hits": [hit.to_dict() for hit in hits],
                "strategy": "信息足够，创建线索并转人工跟进",
            }

        if missing:
            question = self._next_sales_question(missing[0])
            reply = knowledge_text
            if reply:
                reply = f"{reply} {question}".strip()
            else:
                reply = question
            return {
                "reply": reply,
                "intent": "sales_lead",
                "reply_type": "clarify" if missing[0] != "contact" else "collect_lead",
                "next_action": f"collect_{missing[0]}",
                "retrieved_hits": [hit.to_dict() for hit in hits],
                "strategy": "先回答用户关注点，再补销售推进所需字段",
            }

        if state.has_contact and state.lead_id is None:
            state.lead_id = self.storage.create_lead(state)
            state.conversation_outcome = "lead_created"
        reply = f"{knowledge_text} {self._pick_question('schedule')}".strip()
        return {
            "reply": reply,
            "intent": "sales_lead",
            "reply_type": "collect_lead",
            "next_action": "collect_schedule",
            "retrieved_hits": [hit.to_dict() for hit in hits],
            "strategy": "进入安排试听或顾问跟进阶段",
        }

    def _handle_faq(self, user_text: str, state: ConversationState, analysis: dict) -> dict:
        hits = self._retrieve_hits(user_text)
        topic = analysis["topic"] or state.last_topic
        if not hits and topic and SUPPLEMENTAL_FAQ.get(topic):
            answer = f"{self._pick_phrase('warm')} {self._pick_from_list(SUPPLEMENTAL_FAQ[topic])}"
        elif hits:
            answer = self._knowledge_reply(hits, topic)
        else:
            state.clarify_round += 1
            if state.clarify_round > self.settings.clarify_max_rounds:
                answer = "我先不跟您兜圈子。您直接告诉我是想问校区、价格、试听、老师还是售后，我按这一条给您说清楚。"
            else:
                answer = "我先确认一下，您现在更想了解校区、价格、试听、师资，还是售后处理？"
            return {
                "reply": answer,
                "intent": analysis["intent"],
                "reply_type": "clarify",
                "next_action": "narrow_question",
                "retrieved_hits": [],
                "strategy": "没有合适命中时缩小问题范围",
            }

        return {
            "reply": answer,
            "intent": analysis["intent"],
            "reply_type": "answer",
            "next_action": "answer_with_gentle_progression",
            "retrieved_hits": [hit.to_dict() for hit in hits],
            "strategy": "优先使用知识库回复，并避免机械重复",
        }

    def _retrieve_hits(self, user_text: str, preferred_type: str | None = None) -> list[RetrievalHit]:
        return self.rag_service.search_and_rerank(
            user_text,
            top_k=self.settings.top_k,
            top_n=self.settings.top_n,
            preferred_type=preferred_type,
        )

    def _knowledge_reply(self, hits: list[RetrievalHit], topic: str | None) -> str:
        if hits:
            intro = self._pick_phrase("warm" if hits[0].kb_type == "qa" else "empathy")
            return f"{intro} {self._normalize_sentence(hits[0].answer)}".strip()
        if topic and SUPPLEMENTAL_FAQ.get(topic):
            return f"{self._pick_phrase('neutral')} {self._pick_from_list(SUPPLEMENTAL_FAQ[topic])}".strip()
        return ""

    def _fallback_objection_answer(self, objection_type: str) -> str:
        fallback = {
            "busy": "我就说关键一点，先把情况对准再决定要不要继续，不会占您太多时间。",
            "expensive": "费用一定要结合年级、科目和上课方式看，不然报一个空价格意义不大。",
            "far": "如果到校不方便，也可以先了解线上或就近校区的安排。",
            "no_need": "如果您暂时没计划也没关系，我给您留一个最简判断标准，之后需要再联系我就行。",
            "child_unwilling": "孩子不愿意学时，更需要先看原因，光加课不一定有效。",
            "already_have": "已经在学不代表不能再做一次效果判断，重点看现在的问题有没有被真正解决。",
        }
        return fallback.get(objection_type, "我先把实际情况跟您说清楚，再决定下一步。")

    def _objection_next_step(self, objection_type: str, state: ConversationState) -> str:
        if objection_type == "busy":
            return "您如果方便，就一句告诉我是几年级、哪一科，我后面只按这一个点跟您说。"
        if objection_type == "expensive":
            if not state.known_slots.get("grade"):
                return self._pick_question("grade")
            if not state.known_slots.get("subject"):
                return self._pick_question("subject")
            return self._pick_question("contact")
        if objection_type == "far":
            return "您更想了解就近校区，还是先看线上安排？"
        if objection_type == "already_have":
            return "您如果愿意，可以说下孩子现在最卡的是哪一科或哪类问题，我帮您判断现阶段方案够不够。"
        if objection_type == "child_unwilling":
            return "您看孩子是不想学这一科，还是对现在的上课方式排斥？"
        if objection_type == "no_need":
            return "如果后面需要，再直接找我问价格、试听或校区都可以。"
        return ""

    def _identity_next_question(self, state: ConversationState) -> str:
        if state.current_intent == "complaint_after_sales" or state.known_slots.get("order_no"):
            return "您如果是上课后的问题，我现在就按售后流程帮您登记。"
        if not state.known_slots.get("grade"):
            return "您这次主要是想了解孩子课程，还是之前上课问题要处理？"
        if not state.known_slots.get("subject"):
            return "孩子这边现在主要想看哪一科？"
        return "您更想先了解价格、试听，还是离您近的校区？"

    def _missing_sales_fields(self, state: ConversationState, topic: str | None) -> list[str]:
        missing: list[str] = []
        if topic in {"price", "trial", "teacher", "effect", "online"} and not state.known_slots.get("grade"):
            missing.append("grade")
        if topic in {"price", "trial", "teacher", "effect"} and not state.known_slots.get("subject"):
            missing.append("subject")
        if topic in {"cooperation"} and not state.known_slots.get("demand_summary"):
            missing.append("demand_summary")
        if not state.has_contact and not state.contact_refused and topic in {"price", "trial", "teacher", "effect", "cooperation"}:
            missing.append("contact")
        return missing

    def _missing_complaint_fields(self, state: ConversationState) -> list[str]:
        missing: list[str] = []
        if not state.known_slots.get("issue_desc"):
            missing.append("issue_desc")
        if not state.known_slots.get("order_no"):
            missing.append("order_no")
        if not state.has_contact:
            missing.append("contact")
        return missing

    def _next_sales_question(self, field_name: str) -> str:
        if field_name == "contact":
            return self._pick_question("contact")
        return self._pick_question(field_name)

    def _pick_question(self, key: str) -> str:
        return self._pick_from_list(DISCOVERY_QUESTIONS[key])

    def _pick_phrase(self, key: str) -> str:
        return self._pick_from_list(HUMAN_PREFIXES[key])

    def _pick_objection_phrase(self, objection_type: str) -> str:
        return self._pick_from_list(OBJECTION_GUIDES.get(objection_type, HUMAN_PREFIXES["empathy"]))

    def _pick_from_list(self, items: list[str]) -> str:
        index = self._question_cycle_index % len(items)
        self._question_cycle_index += 1
        return items[index]

    def _detect_topic(self, user_text: str) -> str:
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in user_text for keyword in keywords):
                return topic
        return ""

    def _match_objection(self, user_text: str) -> str:
        for objection_type, keywords in OBJECTION_PATTERNS.items():
            if any(keyword in user_text for keyword in keywords):
                return objection_type
        return ""

    def _is_out_of_scope(self, user_text: str) -> bool:
        return any(token in user_text for token in OUT_OF_SCOPE_KEYWORDS)

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).lower()

    def _finalize_reply(self, reply: str, state: ConversationState, analysis: dict) -> str:
        final_reply = re.sub(r"\s+", " ", reply).strip()
        if analysis["repeat_count"] > 0 or final_reply in state.reply_history[-2:]:
            prefix = self._pick_from_list(REPHRASE_PREFIXES)
            if not final_reply.startswith(prefix):
                final_reply = f"{prefix} {final_reply}"
        if analysis["angry"] and "抱歉" not in final_reply:
            final_reply = f"抱歉让您觉得被打扰了。{final_reply}"
        return final_reply

    def _derive_sales_stage(self, state: ConversationState, analysis: dict, next_action: str) -> str:
        if state.ticket_id:
            return "after_sales"
        if state.lead_id:
            return "follow_up"
        if analysis["intent"] == "identity_check":
            return "trust_building"
        if next_action.startswith("collect_"):
            return "qualification"
        if analysis["intent"] == "sales_lead":
            return "discovery"
        return state.sales_stage or "opening"

    def _build_summary(self, state: ConversationState) -> str:
        parts: list[str] = []
        if state.known_slots.get("grade"):
            parts.append(f"年级:{state.known_slots['grade']}")
        if state.known_slots.get("subject"):
            parts.append(f"科目:{state.known_slots['subject']}")
        if state.has_contact:
            contact = state.known_slots.get("contact_phone") or state.known_slots.get("contact_wechat")
            parts.append(f"联系方式:{contact}")
        if state.ticket_id:
            parts.append("已建工单")
        elif state.lead_id:
            parts.append("已建线索")
        if state.last_topic:
            parts.append(f"主题:{state.last_topic}")
        return " | ".join(parts)

    def _build_empty_trace(self, summary: str) -> dict:
        return {
            "summary": summary,
            "workflow": [],
            "knowledge_refs": [],
            "code_refs": self.describe_system()["layers"],
            "files_used": [],
        }

    def _build_trace(
        self,
        *,
        user_text: str,
        state: ConversationState,
        extracted: dict[str, str],
        analysis: dict,
        result: dict,
    ) -> dict:
        code_refs = self.describe_system()["layers"]
        knowledge_refs = self._format_knowledge_refs(result.get("retrieved_hits", []))
        files_used = self._collect_files_used(code_refs, knowledge_refs)
        workflow = [
            self._workflow_step(
                "会话层",
                "读取并更新会话状态",
                f"session_turn={state.turn_index}, sales_stage={state.sales_stage}",
            ),
            self._workflow_step(
                "抽取层",
                "抽取结构化字段",
                "、".join(f"{key}={value}" for key, value in extracted.items()) or "未抽取到新字段",
            ),
            self._workflow_step(
                "路由层",
                f"识别意图={analysis['intent']}",
                "；".join(analysis["reasons"]) or "无",
            ),
            self._workflow_step(
                "知识层",
                "检索知识库/异议库",
                f"命中 {len(knowledge_refs)} 条记录",
            ),
            self._workflow_step(
                "话术层",
                result.get("strategy", "组织回复"),
                f"reply_type={result['reply_type']}, next_action={result['next_action']}",
            ),
            self._workflow_step(
                "业务层",
                "检查线索/工单动作",
                self._business_trace_detail(state, result["next_action"]),
            ),
        ]
        return {
            "summary": self._trace_summary(analysis, result),
            "workflow": workflow,
            "knowledge_refs": knowledge_refs,
            "code_refs": code_refs,
            "files_used": files_used,
            "user_text": user_text,
        }

    def _workflow_step(self, layer_name: str, action: str, detail: str) -> dict:
        ref = next((item for item in self.describe_system()["layers"] if item["layer"] == layer_name), None)
        return {
            "layer": layer_name,
            "action": action,
            "detail": detail,
            "file": ref["file"] if ref else "",
            "file_path": ref["file_path"] if ref else "",
            "file_paths": ref.get("file_paths", []) if ref else [],
            "code": ref["code"] if ref else "",
        }

    def _trace_summary(self, analysis: dict, result: dict) -> str:
        return f"当前走 {analysis['intent']} 分支，策略是：{result.get('strategy', '组织回复')}。"

    def _business_trace_detail(self, state: ConversationState, next_action: str) -> str:
        bits: list[str] = []
        if state.lead_id:
            bits.append(f"lead_id={state.lead_id}")
        if state.ticket_id:
            bits.append(f"ticket_id={state.ticket_id}")
        if state.contact_refused:
            bits.append("用户拒绝留联系方式")
        if not bits:
            bits.append("未触发持久化业务动作")
        bits.append(f"next_action={next_action}")
        return "；".join(bits)

    def _format_knowledge_refs(self, raw_hits: list[dict] | list[RetrievalHit]) -> list[dict]:
        refs: list[dict] = []
        for item in raw_hits:
            hit = item if isinstance(item, dict) else item.to_dict()
            refs.append(
                {
                    "title": hit["title"],
                    "kb_type": hit["kb_type"],
                    "scene": hit.get("scene", ""),
                    "score": hit.get("score", 0),
                    "source_file": hit["source_file"],
                    "path": str((self.settings.knowledge_base_dir / hit["source_file"]).resolve()),
                    "matched_variant": hit.get("matched_variant", ""),
                }
            )
        return refs

    def _collect_files_used(self, code_refs: list[dict], knowledge_refs: list[dict]) -> list[str]:
        seen: list[str] = []
        for item in code_refs:
            for file_path in item.get("file_paths", [item["file_path"]]):
                if file_path not in seen:
                    seen.append(file_path)
        for item in knowledge_refs:
            if item["path"] not in seen:
                seen.append(item["path"])
        return seen

    def _normalize_sentence(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().rstrip("。") + "。"

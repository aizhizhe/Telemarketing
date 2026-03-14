from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any
from uuid import uuid4

from .llm_service import LlmService
from .materials import (
    MaterialRepository,
    clean_text,
    extract_grade,
    extract_subjects,
    extract_wechat,
    get_material_repository,
    split_sentences,
)
from .rules_store import load_rules
from .trace_logger import ConversationTraceLogger


NODE_OPENING = "流程引导库-开场节点"
NODE_DISCOVERY = "流程引导库-挖基础节点"
NODE_PROFESSIONAL = "流程引导库-讲优势节点"
NODE_INVITE = "流程引导库-邀试听节点"

QUESTION_MARKERS = ("?", "？", "吗", "么", "怎么", "为什么", "哪里", "在哪", "有没有", "能不能", "可不可以")
CLOSE_MARKERS = ("没有了", "没了", "没有问题", "没有别的了", "先这样", "再见", "拜拜")
STOP_MARKERS = ("别再打", "不要再打", "别再给我打电话", "不要给我打电话", "别联系了", "不需要", "不用了", "拉黑", "投诉你们", "停一下")
PRIVACY_MARKERS = ("怎么有我电话", "哪来的电话", "谁给你电话", "谁把我的信息", "信息泄露", "隐私", "哪来的信息")
IDENTITY_MARKERS = ("你是谁", "你哪位", "你找我干嘛", "什么机构")
ROBOT_MARKERS = ("你是真人吗", "机器人", "AI吗", "自动回复吗")
ROLE_MARKERS = ("你能帮我做什么", "你能做什么", "你是做什么的")
PING_MARKERS = ("在吗", "在不在", "喂")
COMPLAINT_MARKERS = ("我要投诉", "售后", "退费", "退款", "迟到", "课时", "工单", "订单")
RISKY_MARKERS = ("保证提分", "保分", "保证涨分", "直接承诺", "签保过")
OFFTOPIC_MARKERS = ("天气", "股市", "彩票", "电影", "明星", "八卦")
CONCERN_MARKERS = ("成绩", "掉得厉害", "不太稳", "薄弱", "效果一般", "抵触", "抗拒", "不愿意学", "跟不上", "拉分")
PRICE_MARKERS = ("价格", "收费", "费用", "报价", "贵", "值不值得")
LOCATION_MARKERS = ("在哪", "地址", "校区", "位置", "海淀")
SCHEDULE_MARKERS = ("今晚", "明晚", "今天", "明天", "后天", "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周末", "下周")
TRIAL_MARKERS = ("试听", "试一下课", "试课", "体验课")

OBJECTION_MAP = {
    "效果怀疑": ("行不行", "靠谱吗", "有效果吗", "没效果", "不行"),
    "价格顾虑": ("太贵", "价格", "收费", "学费", "贵不贵", "报价"),
    "没时间": ("没时间", "没空", "太忙", "很忙", "不方便"),
    "孩子抗拒": ("不想学", "抗拒", "抵触", "不愿意"),
    "已报班": ("已经报班", "已经有老师", "在补"),
    "先商量": ("考虑", "商量", "回头再说"),
    "怕被推销": ("推销", "套路", "营销", "广告"),
    "距离问题": ("太远", "不方便过去", "离家远"),
}


def _contains_any(text: str, markers: tuple[str, ...] | list[str]) -> bool:
    return any(marker in text for marker in markers)


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    content = clean_text(text)
    if not content:
        return None
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text))


def _extract_schedule(text: str) -> str | None:
    cleaned = clean_text(text)
    if not cleaned:
        return None
    for marker in SCHEDULE_MARKERS:
        if marker in cleaned:
            return marker
    return None


def _split_reply_sentences(text: str) -> list[str]:
    content = clean_text(text)
    if not content:
        return []
    parts = re.split(r"(?<=[。！？!?])", content)
    return [item.strip() for item in parts if item.strip()]


@dataclass
class ConversationState:
    history: list[dict[str, str]] = field(default_factory=list)
    session_id: str | None = None
    current_node: str = NODE_OPENING
    turn_index: int = 0
    grade: str | None = None
    subject: str | None = None
    wechat_contact: str | None = None
    schedule_preference: str | None = None
    trial_invited: bool = False
    close_prompted: bool = False
    ended: bool = False
    stop_requested: bool = False
    material_gap: bool = False
    last_intent: dict[str, Any] = field(default_factory=dict)
    recent_professional_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ConversationState":
        if not raw:
            return cls()
        state = cls()
        for key in (
            "session_id",
            "current_node",
            "turn_index",
            "grade",
            "subject",
            "wechat_contact",
            "schedule_preference",
            "trial_invited",
            "close_prompted",
            "ended",
            "stop_requested",
            "material_gap",
        ):
            if key in raw:
                setattr(state, key, raw[key])
        state.history = list(raw.get("history") or [])
        state.last_intent = dict(raw.get("last_intent") or {})
        state.recent_professional_ids = list(raw.get("recent_professional_ids") or [])
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def missing_goals(self) -> list[str]:
        ordered = [
            ("grade_collected", bool(self.grade)),
            ("subject_collected", bool(self.subject)),
            ("trial_invited", bool(self.trial_invited)),
            ("wechat_collected", bool(self.wechat_contact)),
        ]
        return [name for name, done in ordered if not done]


class ConversationEngine:
    def __init__(self, materials: MaterialRepository | None = None, llm: LlmService | None = None) -> None:
        self.materials = materials or get_material_repository()
        self.llm = llm or LlmService()
        self.trace_logger = ConversationTraceLogger()

    def start_conversation(self, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        state = ConversationState.from_dict(state_dict)
        if not state.session_id:
            state.session_id = uuid4().hex
        rules = load_rules()
        flow_hits = self.materials.search_flow(node=NODE_OPENING, preferred_tags=["开场白+中性话术"], top_k=5)
        opening = self._pick_opening(flow_hits)
        state.history.append({"role": "assistant", "content": opening})
        return self._result(
            state=state,
            reply=opening,
            user_message=None,
            event_type="start",
            rules=rules,
            extracted={},
            intent={"primary_intent": "opening", "source": "local_fast_path"},
            plan={"node": NODE_OPENING, "missing_goals": state.missing_goals(), "requires_llm": False},
            flow_hits=flow_hits,
            professional_hits=[],
            segments={"guidance_prefix": opening, "professional": "", "guidance_suffix": "", "final_reply": opening},
            llm_calls={},
        )

    def process_turn(self, user_text: str, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        state = ConversationState.from_dict(state_dict)
        if not state.session_id:
            state.session_id = uuid4().hex
        rules = load_rules()
        user_text = _normalize_text(user_text)

        if state.ended:
            return self._result(
                state=state,
                reply="本轮对话已经结束了，您可以重新开始一组新的测试。",
                user_message=user_text,
                event_type="ended_guard",
                rules=rules,
                extracted={},
                intent={"primary_intent": "ended", "source": "local_fast_path"},
                plan={"node": state.current_node, "missing_goals": state.missing_goals(), "requires_llm": False},
                flow_hits=[],
                professional_hits=[],
                segments={},
                llm_calls={},
            )

        if not user_text:
            return self._result(
                state=state,
                reply="您可以直接输入一段家长回复，我会继续沿着链路往下走。",
                user_message=user_text,
                event_type="empty_input",
                rules=rules,
                extracted={},
                intent={"primary_intent": "empty", "source": "local_fast_path"},
                plan={"node": state.current_node, "missing_goals": state.missing_goals(), "requires_llm": False},
                flow_hits=[],
                professional_hits=[],
                segments={},
                llm_calls={},
            )

        state.turn_index += 1
        state.history.append({"role": "user", "content": user_text})

        extracted = self._extract_slots(user_text)
        if extracted["grade"] and not state.grade:
            state.grade = extracted["grade"]
        if extracted["subject"] and not state.subject:
            state.subject = extracted["subject"]
        if extracted["wechat"] and not state.wechat_contact:
            state.wechat_contact = extracted["wechat"]
        if extracted["schedule"] and not state.schedule_preference:
            state.schedule_preference = extracted["schedule"]

        if state.close_prompted and _contains_any(user_text, CLOSE_MARKERS):
            reply = "好的，那我这边就先不继续打扰您了，稍后按刚才确认的信息跟您微信对接，祝孩子学习顺利。"
            state.ended = True
            state.history.append({"role": "assistant", "content": reply})
            return self._result(
                state=state,
                reply=reply,
                user_message=user_text,
                event_type="close",
                rules=rules,
                extracted=extracted,
                intent={"primary_intent": "close_signal", "source": "local_fast_path", "close_signal": True},
                plan={"node": NODE_INVITE, "missing_goals": state.missing_goals(), "requires_llm": False},
                flow_hits=[],
                professional_hits=[],
                segments={"guidance_prefix": reply, "professional": "", "guidance_suffix": "", "final_reply": reply},
                llm_calls={},
            )

        base_intent = self._classify_turn(user_text, extracted)
        plan = self._plan_turn(state, base_intent)

        flow_hits = self.materials.search_flow(node=plan["node"], preferred_tags=plan["preferred_flow_tags"], top_k=6)
        professional_hits: list[dict[str, Any]] = []
        if plan["requires_professional"]:
            professional_hits, state.material_gap = self.materials.search_professional(
                query=user_text,
                objection_label=base_intent.get("objection_label"),
                professional_topics=list(base_intent.get("professional_topics") or []),
                grade=state.grade,
                subject=state.subject,
                top_k=6,
            )
            professional_hits = self._dedupe_recent_professional_hits(professional_hits, state)
        else:
            state.material_gap = False

        llm_calls: dict[str, Any] = {}
        if self._needs_llm(base_intent, plan):
            combined = self._combined_intent_and_guidance(
                user_text=user_text,
                state=state,
                rules=rules,
                base_intent=base_intent,
                plan=plan,
                flow_hits=flow_hits,
                professional_hits=professional_hits,
            )
            llm_calls["combined"] = combined["llm_call"]
            intent = combined["intent"]
            prefix = combined["prefix"]
            suffix = combined["suffix"]
        else:
            intent = self._local_intent(base_intent)
            prefix, suffix = self._local_guidance(
                user_text=user_text,
                state=state,
                base_intent=base_intent,
                plan=plan,
                professional_hits=professional_hits,
            )

        prefix, suffix = self._refine_guidance(
            user_text=user_text,
            state=state,
            base_intent=base_intent,
            prefix=prefix,
            suffix=suffix,
        )
        prefix = self._sanitize_guidance_prefix(prefix=prefix, professional_hits=professional_hits, base_intent=base_intent)

        state.last_intent = intent
        professional = self._build_professional_segment(professional_hits, state, prefix=prefix)
        reply = self._assemble_reply(prefix=prefix, professional=professional, suffix=suffix)

        if "试听" in reply or "体验" in reply:
            state.trial_invited = True
        state.current_node = plan["node"]
        if base_intent.get("fast_case") == "stop_call":
            state.stop_requested = True
            state.ended = True
        if not state.missing_goals() and not state.close_prompted:
            if "您这边还有别的问题吗" not in reply:
                reply = f"{reply} 您这边还有别的问题吗？".strip()
            state.close_prompted = True

        state.history.append({"role": "assistant", "content": reply})
        state.history = state.history[-14:]

        return self._result(
            state=state,
            reply=reply,
            user_message=user_text,
            event_type="turn",
            rules=rules,
            extracted=extracted,
            intent=intent,
            plan={**plan, "requires_llm": self._needs_llm(base_intent, plan)},
            flow_hits=flow_hits,
            professional_hits=professional_hits,
            segments={"guidance_prefix": prefix, "professional": professional, "guidance_suffix": suffix, "final_reply": reply},
            llm_calls=llm_calls,
        )

    def _extract_slots(self, user_text: str) -> dict[str, str | None]:
        subjects = extract_subjects(user_text)
        return {
            "grade": extract_grade(user_text),
            "subject": subjects[0] if subjects else None,
            "wechat": extract_wechat(user_text),
            "schedule": _extract_schedule(user_text),
        }

    def _classify_turn(self, user_text: str, extracted: dict[str, str | None]) -> dict[str, Any]:
        objection_label = None
        for label, markers in OBJECTION_MAP.items():
            if _contains_any(user_text, markers):
                objection_label = label
                break

        fast_case = None
        if _contains_any(user_text, STOP_MARKERS):
            fast_case = "stop_call"
        elif _contains_any(user_text, PRIVACY_MARKERS):
            fast_case = "privacy"
        elif _contains_any(user_text, COMPLAINT_MARKERS):
            fast_case = "complaint"
        elif _contains_any(user_text, RISKY_MARKERS):
            fast_case = "risky"
        elif _contains_any(user_text, ROLE_MARKERS):
            fast_case = "role_explain"
        elif _contains_any(user_text, IDENTITY_MARKERS):
            fast_case = "identity"
        elif _contains_any(user_text, ROBOT_MARKERS):
            fast_case = "robot"
        elif _contains_any(user_text, TRIAL_MARKERS) and not objection_label:
            fast_case = "trial_request"
        elif user_text in PING_MARKERS:
            fast_case = "small_talk"
        elif _contains_any(user_text, OFFTOPIC_MARKERS):
            fast_case = "off_topic"
        elif self._is_slot_only_turn(user_text, extracted, objection_label):
            fast_case = "slot_only"

        professional_topics: list[str] = []
        if _contains_any(user_text, PRICE_MARKERS):
            professional_topics.append("费用")
        if _contains_any(user_text, LOCATION_MARKERS):
            professional_topics.extend(["校区", "线上"])
        if _contains_any(user_text, TRIAL_MARKERS):
            professional_topics.extend(["试听", "老师", "匹配"])
        if _contains_any(user_text, CONCERN_MARKERS):
            professional_topics.extend(["成绩", "提分", "薄弱"])
        for topic in ("效果", "试听", "老师", "线上", "微信", "投诉", "隐私"):
            if topic in user_text:
                professional_topics.append(topic)
        if extracted["subject"]:
            professional_topics.append(extracted["subject"])
        professional_topics = list(dict.fromkeys(professional_topics))

        return {
            "primary_intent": fast_case or ("objection" if objection_label else "question" if _contains_any(user_text, QUESTION_MARKERS) else "statement"),
            "objection_label": objection_label,
            "professional_topics": professional_topics,
            "fast_case": fast_case,
            "close_signal": _contains_any(user_text, CLOSE_MARKERS),
            "question_like": _contains_any(user_text, QUESTION_MARKERS) or _contains_any(user_text, PRICE_MARKERS) or _contains_any(user_text, LOCATION_MARKERS),
            "needs_professional_answer": bool(
                objection_label
                or _contains_any(user_text, CONCERN_MARKERS)
                or _contains_any(user_text, PRICE_MARKERS)
                or _contains_any(user_text, LOCATION_MARKERS)
                or _contains_any(user_text, TRIAL_MARKERS)
                or "老师" in user_text
                or "效果" in user_text
            ),
        }

    def _plan_turn(self, state: ConversationState, base_intent: dict[str, Any]) -> dict[str, Any]:
        fast_case = base_intent.get("fast_case")
        missing = state.missing_goals()
        node = NODE_DISCOVERY
        preferred_tags: list[str] = []
        requires_professional = False

        if fast_case in {"stop_call", "privacy", "complaint", "risky"}:
            node = NODE_PROFESSIONAL
            requires_professional = fast_case in {"privacy", "complaint", "risky"}
        elif fast_case in {"identity", "robot", "role_explain", "small_talk"}:
            node = NODE_DISCOVERY
        elif fast_case == "trial_request":
            node = NODE_INVITE
            preferred_tags = ["邀约加微+试听价值", "邀约加微+选择式邀约"]
            requires_professional = True
        elif base_intent.get("needs_professional_answer"):
            node = NODE_PROFESSIONAL
            preferred_tags = ["激发危机感+共情", "分析价值+问题分析", "分析价值+价值塑造+信任背书"]
            requires_professional = True
        elif not state.grade:
            node = NODE_DISCOVERY
            preferred_tags = ["引导发问+年级提问"]
        elif not state.subject:
            node = NODE_DISCOVERY
            preferred_tags = ["引导发问+薄弱科目提问"]
        elif not state.trial_invited:
            node = NODE_INVITE
            preferred_tags = ["邀约加微+试听价值", "邀约加微+选择式邀约"]
        elif not state.wechat_contact:
            node = NODE_INVITE
            preferred_tags = ["邀约加微+加微理由"]
        else:
            node = NODE_INVITE
            preferred_tags = ["礼貌收尾+确认完成"]

        if not missing:
            node = NODE_INVITE
            preferred_tags = ["礼貌收尾+确认完成"]
            requires_professional = False

        return {
            "node": node,
            "preferred_flow_tags": preferred_tags,
            "missing_goals": missing,
            "requires_professional": requires_professional,
            "next_goal": missing[0] if missing else "polite_close",
        }

    def _needs_llm(self, base_intent: dict[str, Any], plan: dict[str, Any]) -> bool:
        if base_intent.get("fast_case") in {"stop_call", "privacy", "complaint", "risky", "identity", "robot", "role_explain", "small_talk", "slot_only", "off_topic", "trial_request"}:
            return False
        return bool(plan["requires_professional"])

    def _combined_intent_and_guidance(
        self,
        *,
        user_text: str,
        state: ConversationState,
        rules: dict[str, str],
        base_intent: dict[str, Any],
        plan: dict[str, Any],
        flow_hits: list[dict[str, Any]],
        professional_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{rules['global_objective']}\n"
                    f"{rules['intent_system_prompt']}\n"
                    f"{rules['guidance_system_prompt']}\n"
                    f"{rules['retrieval_policy']}\n"
                    f"{rules['assembly_policy']}\n"
                    f"{rules['closing_policy']}\n"
                    "只输出一个 JSON，字段必须包含："
                    "primary_intent, objection_label, emotion_needed, professional_needed, flow_break, return_to_flow,"
                    "close_signal, tone_hint, prefix, suffix_question, style_notes。"
                    "要求：前缀只做简短共情和转场，控制在 18 个字以内，不要重复专业正文，不要在前缀里提前讲价格、校区、老师、试听、提分细节；"
                    "后缀只做拉回目标或礼貌收尾；"
                    "语气像真人电话销售，不要空话，不要复读。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "latest_user_message": user_text,
                        "recent_history": state.history[-6:],
                        "known_slots": {
                            "grade": state.grade,
                            "subject": state.subject,
                            "wechat_contact": state.wechat_contact,
                            "schedule_preference": state.schedule_preference,
                            "trial_invited": state.trial_invited,
                        },
                        "missing_goals": state.missing_goals(),
                        "provisional_intent": base_intent,
                        "plan": plan,
                        "flow_reference": [hit["content"] for hit in flow_hits[:3]],
                        "professional_reference": [
                            {
                                "title": hit.get("title"),
                                "category": hit.get("category"),
                                "content": split_sentences(str(hit.get("content") or ""), limit=1),
                            }
                            for hit in professional_hits[:3]
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        llm_call = self.llm.chat(
            stage="combined_intent_guidance",
            messages=messages,
            temperature=0.2,
            max_tokens=320,
            model=self.llm.settings.fast_chat_model,
        )
        parsed = _safe_json_parse(llm_call["response_text"]) or {}
        intent = {
            "primary_intent": clean_text(parsed.get("primary_intent")) or base_intent["primary_intent"],
            "objection_label": clean_text(parsed.get("objection_label")) or base_intent.get("objection_label"),
            "emotion_needed": bool(parsed.get("emotion_needed", False)),
            "professional_needed": bool(parsed.get("professional_needed", plan["requires_professional"])),
            "flow_break": bool(parsed.get("flow_break", False)),
            "return_to_flow": bool(parsed.get("return_to_flow", True)),
            "close_signal": bool(parsed.get("close_signal", False)),
            "tone_hint": clean_text(parsed.get("tone_hint")) or "真人、专业、销售推进",
            "style_notes": clean_text(parsed.get("style_notes")),
            "source": "single_llm_call",
        }
        return {
            "intent": intent,
            "prefix": clean_text(parsed.get("prefix")) or self._fallback_prefix(base_intent, user_text=user_text, state=state),
            "suffix": clean_text(parsed.get("suffix_question")) or self._default_suffix(state, base_intent),
            "llm_call": llm_call,
        }

    def _local_intent(self, base_intent: dict[str, Any]) -> dict[str, Any]:
        return {
            "primary_intent": base_intent["primary_intent"],
            "objection_label": base_intent.get("objection_label"),
            "emotion_needed": base_intent["primary_intent"] in {"privacy", "complaint"},
            "professional_needed": bool(base_intent.get("fast_case") == "trial_request"),
            "flow_break": False,
            "return_to_flow": base_intent.get("fast_case") not in {"stop_call", "privacy", "complaint", "risky"},
            "close_signal": bool(base_intent.get("close_signal")),
            "tone_hint": "local_fast_path",
            "source": "local_fast_path",
        }

    def _local_guidance(
        self,
        *,
        user_text: str,
        state: ConversationState,
        base_intent: dict[str, Any],
        plan: dict[str, Any],
        professional_hits: list[dict[str, Any]],
    ) -> tuple[str, str]:
        fast_case = base_intent.get("fast_case")
        if fast_case == "stop_call":
            return "抱歉打扰您了，我这边马上给您标记停联，不会再继续打扰。", ""
        if fast_case == "privacy":
            privacy_count = sum(1 for item in state.history if item["role"] == "user" and _contains_any(item["content"], PRIVACY_MARKERS))
            if professional_hits and privacy_count <= 1:
                return "", ""
            if privacy_count > 1:
                return "我理解您是在确认这个点，您如果不希望继续联系，我现在就能给您停联；如果愿意继续，我只说明来电目的，绝不会继续打扰。", ""
            return "理解您的顾虑，这类问题家长都会比较敏感。", ""
        if fast_case == "complaint":
            prefix = "" if professional_hits else "抱歉给您带来不好的体验，这类问题我先按售后登记处理。"
            suffix = "您把方便接收回复的手机号或微信留一下，我这边给您登记售后处理。" if not state.wechat_contact else "我这边已经记下联系方式了，会尽快给您登记处理。"
            return prefix, suffix
        if fast_case == "risky":
            return ("", self._default_suffix(state, base_intent)) if professional_hits else ("这个我不能给您做保分承诺。", self._default_suffix(state, base_intent))
        if fast_case == "identity":
            count = sum(1 for item in state.history if item["role"] == "user" and _contains_any(item["content"], IDENTITY_MARKERS))
            if count <= 1:
                return "您好，我这边是北文教育课程咨询这边，主要是想了解下孩子最近的学习情况，看看有没有适合的提升方向。", ""
            return "您好，我这边是北文教育课程咨询这边，这通电话主要是先判断孩子最近有没有哪科更需要重点提升，合适的话我再给您安排试听和资料参考。", "要是您方便，我先了解下孩子最近哪科更需要提升？"
        if fast_case == "robot":
            return "不是机器人，我这边是真人顾问在跟您沟通，主要负责孩子课程咨询和试听安排。", ""
        if fast_case == "trial_request":
            return "可以的，我这边先作为课程顾问给您记下，后面会按孩子的年级和学科去匹配更合适的老师。", self._default_suffix(state, base_intent)
        if fast_case == "role_explain":
            return "我这边主要是帮家长判断孩子当前学习问题、匹配合适老师，再安排试听和后续资料。", "要是您方便，我先了解下孩子最近哪科更需要提升？"
        if fast_case == "small_talk":
            return "在的，我这边是课程顾问。", "您是想先了解孩子哪一科，还是先问试听安排？"
        if fast_case == "off_topic":
            return "这个话题我知道，不过我还是先围绕孩子学习这件事帮您把重点捋清楚。", self._default_suffix(state, base_intent)
        if fast_case == "slot_only":
            return self._slot_ack(state), self._default_suffix(state, base_intent)
        if not plan["missing_goals"]:
            return "好的，那我这边就按刚才确认的信息给您安排，稍后微信上和您对接。", "您这边还有别的问题吗？"
        return self._fallback_prefix(base_intent, user_text=user_text, state=state), self._default_suffix(state, base_intent)

    def _slot_ack(self, state: ConversationState) -> str:
        if state.schedule_preference:
            return f"好的，{state.schedule_preference}我先给您记上。"
        if state.wechat_contact:
            variants = [
                "好的，微信我先记下了。",
                "行，联系方式我先给您记上。",
            ]
            return variants[state.turn_index % len(variants)]
        if state.grade and not state.subject:
            return "这个年级我先记下了。"
        if state.subject and not state.trial_invited:
            return "这科我有数了。"
        if state.grade and state.subject and not state.wechat_contact:
            variants = [
                "这个情况我先给您记下来。",
                "行，我先按您刚说的情况记着。",
            ]
            return variants[state.turn_index % len(variants)]
        return "好的，我先记一下。"

    def _build_professional_segment(self, hits: list[dict[str, Any]], state: ConversationState, *, prefix: str = "") -> str:
        if not hits:
            return ""
        prefix_norm = _normalize_text(prefix)
        candidate_hits = [hit for hit in hits if hit["material_id"] not in state.recent_professional_ids] or hits
        chosen_hit = candidate_hits[0]
        chosen_segment = ""
        for hit in candidate_hits:
            segment = self._slice_professional_content(str(hit.get("content") or ""))
            sentence_norms = [_normalize_text(item) for item in _split_reply_sentences(segment)]
            if sentence_norms and all(item == prefix_norm or item in prefix_norm or prefix_norm in item for item in sentence_norms):
                continue
            chosen_hit = hit
            chosen_segment = segment
            break
        if not chosen_segment:
            chosen_segment = self._slice_professional_content(str(chosen_hit.get("content") or ""))
        state.recent_professional_ids.append(chosen_hit["material_id"])
        state.recent_professional_ids = state.recent_professional_ids[-6:]
        return chosen_segment

    def _slice_professional_content(self, text: str) -> str:
        limit = 1
        if any(keyword in text for keyword in ("停联", "停止联系", "抗拒", "原因", "方式", "重点", "没时间")):
            limit = 2
        segment = split_sentences(text, limit=limit)
        max_len = 170
        if any(keyword in text for keyword in ("费用", "价格", "报价")):
            max_len = 115
        elif any(keyword in text for keyword in ("校区", "位置", "海淀", "线上")):
            max_len = 130
        if len(segment) > max_len:
            segment = segment[:max_len].rstrip("，,；; ") + "。"
        return segment

    def _assemble_reply(self, *, prefix: str, professional: str, suffix: str) -> str:
        sentences: list[str] = []
        seen: list[str] = []
        for part in (prefix, professional, suffix):
            for sentence in _split_reply_sentences(part):
                normalized = _normalize_text(sentence)
                if not normalized:
                    continue
                if any(normalized == item or normalized in item or item in normalized for item in seen):
                    continue
                sentences.append(sentence)
                seen.append(normalized)
        return _normalize_text(" ".join(sentences))

    def _default_suffix(self, state: ConversationState, base_intent: dict[str, Any]) -> str:
        if base_intent.get("fast_case") == "stop_call":
            return ""
        if base_intent.get("fast_case") == "privacy":
            return ""
        if not state.grade:
            return "您家孩子现在几年级了？"
        if not state.subject:
            return "孩子目前最需要先提升的是哪一科？"
        if not state.trial_invited:
            if state.schedule_preference:
                return f"要不我先按您说的{state.schedule_preference}这个时间方向，给孩子约一节针对性的试听，您看可以吗？"
            return "要不我先给孩子约一节针对性的试听，您看这周还是下周更方便？"
        if not state.wechat_contact:
            return "您微信号方便留一下吗？我把对应资料和试听信息发您。"
        return "您这边还有别的问题吗？"

    def _refine_guidance(
        self,
        *,
        user_text: str,
        state: ConversationState,
        base_intent: dict[str, Any],
        prefix: str,
        suffix: str,
    ) -> tuple[str, str]:
        if base_intent.get("objection_label") == "没时间":
            prefix = "理解，您现在忙，我就抓关键、说重点。"
        elif base_intent.get("objection_label") == "孩子抗拒":
            prefix = "这种情况很常见，我能理解，先找到孩子抵触的原因，再换孩子更容易接受的方式。"
        elif base_intent.get("fast_case") == "trial_request":
            prefix = "可以的，我这边先作为课程顾问给您记下，后面会按孩子的年级和学科去匹配更合适的老师。"
        elif _contains_any(user_text, LOCATION_MARKERS):
            repeat_count = sum(
                1
                for item in state.history
                if item["role"] == "user" and _contains_any(item["content"], LOCATION_MARKERS)
            )
            if repeat_count >= 2:
                prefix = "我给您说清楚一点，我们主要校区在海淀人大附附近，也可以先走线上一对一。"

        if base_intent.get("objection_label") == "没时间" and len(clean_text(suffix)) > 28:
            suffix = "您先告诉我孩子几年级、哪科最需要提一下就行。"
        return prefix, suffix

    def _sanitize_guidance_prefix(
        self,
        *,
        prefix: str,
        professional_hits: list[dict[str, Any]],
        base_intent: dict[str, Any],
    ) -> str:
        cleaned = clean_text(prefix)
        if not cleaned or not professional_hits:
            return cleaned
        if any(keyword in cleaned for keyword in ("海淀", "线上", "校区", "费用", "报价", "试听", "老师", "匹配", "提分", "薄弱")):
            objection = base_intent.get("objection_label")
            if objection == "价格顾虑":
                return "理解，先问费用很正常。"
            if objection == "效果怀疑":
                return "您先确认靠不靠谱，这个很正常。"
            if objection == "孩子抗拒":
                return "孩子有抵触情绪，这种情况很常见。"
            topics = base_intent.get("professional_topics") or []
            if "校区" in topics or "线上" in topics:
                return "您先把位置问清楚，这个很正常。"
            if "试听" in topics:
                return "先试听再决定，这个思路没问题。"
            return "这个问题我先跟您说清楚。"
        return cleaned

    def _fallback_prefix(self, base_intent: dict[str, Any], *, user_text: str = "", state: ConversationState | None = None) -> str:
        if base_intent.get("fast_case") == "trial_request":
            return "可以的，我这边先作为课程顾问给您记下，后面会按孩子的年级和学科去匹配更合适的老师。"
        if base_intent.get("objection_label") == "没时间":
            return "理解，您现在忙，我就抓关键、说重点。"
        if base_intent.get("objection_label") == "孩子抗拒":
            return "这种情况很常见，我能理解，先找到孩子抵触的原因，再换孩子更容易接受的方式。"
        if _contains_any(user_text, LOCATION_MARKERS) and state:
            repeat_count = sum(
                1
                for item in state.history
                if item["role"] == "user" and _contains_any(item["content"], LOCATION_MARKERS)
            )
            if repeat_count >= 2:
                return "我给您说清楚一点，我们主要校区在海淀人大附附近，也可以先走线上一对一。"
        if base_intent.get("objection_label"):
            return "您的顾虑我能理解，我先把关键点跟您说清楚。"
        if base_intent.get("question_like"):
            return "这个问题问得很实际。"
        return "明白，我接着帮您往下梳理。"

    def _dedupe_recent_professional_hits(self, hits: list[dict[str, Any]], state: ConversationState) -> list[dict[str, Any]]:
        fresh = [hit for hit in hits if hit["material_id"] not in state.recent_professional_ids]
        return fresh or hits

    def _pick_opening(self, flow_hits: list[dict[str, Any]]) -> str:
        if not flow_hits:
            return "您好，我们这边是做孩子一对一学科提升的，想先了解下孩子现在几年级、哪科更需要提升？"
        return clean_text(flow_hits[0].get("content"))

    def _is_slot_only_turn(self, user_text: str, extracted: dict[str, str | None], objection_label: str | None) -> bool:
        if objection_label or _contains_any(user_text, QUESTION_MARKERS):
            return False
        if _contains_any(user_text, CONCERN_MARKERS) or _contains_any(user_text, PRICE_MARKERS) or _contains_any(user_text, LOCATION_MARKERS):
            return False
        meaningful_slots = [value for value in extracted.values() if value]
        return bool(meaningful_slots and len(user_text) <= 36)

    def _result(
        self,
        *,
        state: ConversationState,
        reply: str,
        user_message: str | None,
        event_type: str,
        rules: dict[str, str],
        extracted: dict[str, Any],
        intent: dict[str, Any],
        plan: dict[str, Any],
        flow_hits: list[dict[str, Any]],
        professional_hits: list[dict[str, Any]],
        segments: dict[str, Any],
        llm_calls: dict[str, Any],
    ) -> dict[str, Any]:
        llm_summary = self._summarize_llm_usage(intent=intent, llm_calls=llm_calls)
        debug = {
            "active_rules": rules,
            "slot_extract": extracted,
            "intent": intent,
            "plan": plan,
            "flow_hits": flow_hits[:4],
            "professional_hits": professional_hits[:4],
            "segments": segments,
            "llm_calls": llm_calls,
            "llm_summary": llm_summary,
            "material_stats": self.materials.stats(),
        }
        trace_info = self.trace_logger.log_event(
            session_id=state.session_id or "unknown",
            event_type=event_type,
            user_message=user_message,
            assistant_reply=reply,
            state=state.to_dict(),
            debug=debug,
        )
        debug["trace"] = trace_info
        return {
            "reply": reply,
            "state": state.to_dict(),
            "debug": debug,
        }

    def _summarize_llm_usage(self, *, intent: dict[str, Any], llm_calls: dict[str, Any]) -> dict[str, Any]:
        if llm_calls.get("combined"):
            call = llm_calls["combined"]
            return {
                "used": True,
                "mode": "combined",
                "stage": call.get("stage"),
                "model": call.get("model"),
                "reason": "本轮走了单次合并 LLM 调用，用于意图识别和引导话术生成。",
            }
        source = intent.get("source") or "local_fast_path"
        reasons = {
            "local_fast_path": "本轮命中了本地快速路径，没有调用 LLM。",
            "single_llm_call": "本轮走了单次合并 LLM 调用。",
        }
        return {
            "used": False,
            "mode": source,
            "stage": None,
            "model": None,
            "reason": reasons.get(source, "本轮没有触发 LLM。"),
        }

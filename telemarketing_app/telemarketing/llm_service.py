from __future__ import annotations

import json
import re
from typing import Any

import requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .models import ConversationState, RetrievalHit
from .settings import Settings, get_settings


class TelemarketingLLMService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._disabled_reason = ""
        self._client = None
        if (
            OpenAI is not None
            and self.settings.llm_provider in {"dashscope", "openai_compatible"}
            and self.settings.llm_api_key
        ):
            self._client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_enabled and self.settings.llm_api_key and not self._disabled_reason)

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    def analyze_turn(
        self,
        *,
        user_text: str,
        state: ConversationState,
        extracted: dict[str, str],
        fallback: dict,
        hits: list[RetrievalHit],
    ) -> dict | None:
        if not self.enabled:
            return None
        prompt = {
            "task": "classify_turn",
            "user_text": user_text,
            "recent_user_history": state.user_history[-4:],
            "recent_reply_history": state.reply_history[-3:],
            "known_slots": state.known_slots,
            "extracted_slots": extracted,
            "current_intent": state.current_intent,
            "last_topic": state.last_topic,
            "fallback_analysis": fallback,
            "knowledge_hits": [
                {
                    "title": hit.title,
                    "kb_type": hit.kb_type,
                    "scene": hit.scene,
                    "matched_variant": hit.matched_variant,
                }
                for hit in hits[:3]
            ],
            "allowed_intents": [
                "faq_consult",
                "sales_lead",
                "identity_check",
                "privacy_concern",
                "stop_request",
                "complaint_after_sales",
                "small_talk",
                "out_of_scope",
                "risky_request",
                "need_clarify",
            ],
            "rules": [
                "identity_check covers who are you / what institution / why are you calling / are you a robot",
                "privacy_concern covers how did you get my phone / who leaked my information / repeated calls concern",
                "stop_request covers do not call again / do not contact again / this is harassment",
                "risky_request covers guarantee score improvement or illegal promises",
                "sales_lead means the user is consulting course, trial, price, teacher, arrangement or has provided child/contact info",
                "Return JSON only",
            ],
            "json_schema": {
                "intent": "string",
                "topic": "string",
                "objection_type": "string",
                "reasons": ["string"],
            },
        }
        text = self._chat(
            system_prompt=(
                "You are a strict intent classifier for a K12 telemarketing assistant. "
                "Return compact JSON only. Do not include markdown."
            ),
            user_payload=prompt,
            temperature=0.0,
            max_tokens=220,
        )
        parsed = self._parse_json(text)
        if not parsed:
            return None
        return {
            "intent": parsed.get("intent") or fallback["intent"],
            "topic": parsed.get("topic") or fallback["topic"],
            "objection_type": parsed.get("objection_type") or fallback["objection_type"],
            "repeat_count": fallback["repeat_count"],
            "angry": fallback["angry"],
            "positive": fallback["positive"],
            "reasons": parsed.get("reasons") or fallback["reasons"],
        }

    def rewrite_reply(
        self,
        *,
        user_text: str,
        state: ConversationState,
        analysis: dict,
        draft_reply: str,
        hits: list[dict] | list[RetrievalHit],
    ) -> str | None:
        if not self.enabled:
            return None
        normalized_hits: list[dict[str, Any]] = []
        for hit in hits[:3]:
            if isinstance(hit, RetrievalHit):
                normalized_hits.append(
                    {
                        "title": hit.title,
                        "answer": hit.answer,
                        "scene": hit.scene,
                        "kb_type": hit.kb_type,
                    }
                )
            else:
                normalized_hits.append(hit)
        prompt = {
            "task": "rewrite_reply",
            "user_text": user_text,
            "intent": analysis["intent"],
            "topic": analysis["topic"],
            "known_slots": state.known_slots,
            "draft_reply": draft_reply,
            "knowledge_hits": normalized_hits,
            "requirements": [
                "Sound like a professional human K12 sales consultant",
                "Be concise and natural, no robotic repetition",
                "Do not invent facts outside knowledge hits and draft reply",
                "If user asks identity/privacy/stop contact, prioritize trust and boundaries over sales push",
                "If the user has clear purchase intent, keep one soft sales next step",
                "Return plain text only",
            ],
        }
        text = self._chat(
            system_prompt=(
                "You rewrite telemarketing replies into natural Chinese for a K12 education consultant. "
                "Return plain text only."
            ),
            user_payload=prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=280,
        )
        if not text:
            return None
        cleaned = text.strip()
        if len(cleaned) < 6:
            return None
        return cleaned

    def _chat(self, *, system_prompt: str, user_payload: dict, temperature: float, max_tokens: int) -> str | None:
        if not self.enabled:
            return None
        if self.settings.llm_provider in {"dashscope", "openai_compatible"}:
            return self._chat_openai_compatible(system_prompt, user_payload, temperature, max_tokens)
        return self._chat_http(system_prompt, user_payload, temperature, max_tokens)

    def _chat_openai_compatible(self, system_prompt: str, user_payload: dict, temperature: float, max_tokens: int) -> str | None:
        if self._client is None:
            self._disabled_reason = "client_not_available"
            return None
        try:
            response = self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            message = str(exc)
            self._disabled_reason = "invalid_credentials" if "401" in message or "invalid" in message.lower() else "request_failed"
            return None
        try:
            return (response.choices[0].message.content or "").strip()
        except Exception:  # pragma: no cover
            return None

    def _chat_http(self, system_prompt: str, user_payload: dict, temperature: float, max_tokens: int) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        try:
            response = requests.post(
                self.settings.llm_base_url,
                headers=headers,
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover
            self._disabled_reason = f"network_error:{exc}"
            return None
        if response.status_code == 401:
            self._disabled_reason = "invalid_credentials"
            return None
        if response.status_code >= 400:
            self._disabled_reason = f"http_{response.status_code}"
            return None
        try:
            data = response.json()
        except Exception:  # pragma: no cover
            return None
        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()

    def _parse_json(self, text: str | None) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

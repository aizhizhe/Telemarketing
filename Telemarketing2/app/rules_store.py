from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .config import PROJECT_ROOT


RULES_PATH = PROJECT_ROOT / "config" / "rules.json"


DEFAULT_RULES: dict[str, str] = {
    "global_objective": (
        "你是电话招生单 Agent 的链路大脑。总目标是完成四件事：收集微信号、收集年级、收集学科、完成试听邀约。"
        "即使用户打断流程，也要先处理当前问题，再自然拉回未完成目标。"
    ),
    "intent_system_prompt": (
        "你只做意图分析，不直接回复家长。必须输出 JSON。"
        "判断当前轮是否需要情绪引导、是否需要专业答复、是否打断主流程、是否需要拉回流程，"
        "并识别 objection_label、professional_topics、close_signal、return_to_flow。"
        "禁止输出 JSON 之外的内容。"
    ),
    "guidance_system_prompt": (
        "你只生成引导性话术，不提供专业事实。"
        "引导性话术可以做共情、降压、转场、提出下一步问题、礼貌收尾，"
        "但不能编造师资、价格、效果、课程等专业信息。"
        "如果后面有专业素材，前缀负责安抚和承接，后缀负责把对话拉回流程。"
        "必须输出 JSON，字段为 prefix、suffix_question、tone。"
    ),
    "retrieval_policy": (
        "专业性内容只能来自 Data；流程推进参考只能来自 Data2。"
        "当家长问专业问题或提出异议时，先命中 Data 的专业素材；"
        "当需要回到流程时，参考 Data2 当前节点素材和规则。"
        "如果命中分低，标记素材缺口。"
    ),
    "assembly_policy": (
        "最终回复按顺序拼接：引导前缀 -> 专业素材 -> 拉回流程的问题。"
        "如果当前轮不需要专业素材，可以只输出引导前缀和拉回问题。"
        "回复尽量控制在 3 句话内，先回应家长，再推进目标。"
    ),
    "closing_policy": (
        "只有在微信号、年级、学科、试听邀约全部完成后，才能进入礼貌收尾。"
        "礼貌收尾时优先询问“您这边还有别的问题吗”，没有新问题再结束。"
    ),
}


def _merged_rules(data: dict[str, str] | None = None) -> dict[str, str]:
    merged = deepcopy(DEFAULT_RULES)
    if data:
        for key, value in data.items():
            if key in merged and isinstance(value, str):
                merged[key] = value
    return merged


def ensure_rules_file() -> None:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RULES_PATH.exists():
        return
    RULES_PATH.write_text(json.dumps(DEFAULT_RULES, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rules() -> dict[str, str]:
    ensure_rules_file()
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return _merged_rules(raw)


def save_rules(updated: dict[str, str]) -> dict[str, str]:
    merged = _merged_rules(updated)
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged

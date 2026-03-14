from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine import ConversationEngine


TRAIN_SCENARIOS = PROJECT_ROOT.parent / "Telemarketing" / "Train" / "benchmark_scenarios_500.jsonl"
REPORT_DIR = PROJECT_ROOT / "qa" / "reports"

LEAD_CATEGORIES = {"price", "trial", "busy", "expensive", "already_have", "child_unwilling", "small_talk", "effect", "repeat_question"}
PROFESSIONAL_CATEGORIES = {"price", "expensive", "already_have", "child_unwilling", "effect", "repeat_question", "trial"}

GRADE_SUBJECT_CHOICES = [
    ("初二", "数学"),
    ("初一", "英语"),
    ("小学五年级", "语文"),
    ("高一", "物理"),
    ("初三", "化学"),
]
CONTACT_CHOICES = [
    "微信是 abc12345",
    "加我微信 wxmath2026",
    "手机号同微信，13800138000",
]
SCHEDULE_CHOICES = ["这周六下午可以", "下周二晚上方便", "明晚七点后有空"]


@dataclass
class ScenarioResult:
    scenario: dict[str, Any]
    conversation: list[dict[str, Any]]
    state: dict[str, Any]
    checks: list[dict[str, Any]]
    passed_checks: int
    total_checks: int
    accuracy: float
    elapsed_seconds: float


def load_scenarios() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TRAIN_SCENARIOS.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_sample(scenarios: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        by_category[scenario.get("category", "unknown")].append(scenario)

    total = len(scenarios)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for category, items in sorted(by_category.items()):
        rng.shuffle(items)
        quota = max(1, round(sample_size * len(items) / total))
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    rng.shuffle(leftovers)
    deduped = {item["scenario_id"]: item for item in selected}
    for item in leftovers:
        if len(deduped) >= sample_size:
            break
        deduped.setdefault(item["scenario_id"], item)
    return list(deduped.values())[:sample_size]


def render_template_turn(text: str, idx: int) -> str:
    grade, subject = GRADE_SUBJECT_CHOICES[idx % len(GRADE_SUBJECT_CHOICES)]
    contact = CONTACT_CHOICES[idx % len(CONTACT_CHOICES)]
    schedule = SCHEDULE_CHOICES[idx % len(SCHEDULE_CHOICES)]
    rendered = text.replace("{grade}", grade).replace("{subject}", subject)
    rendered = rendered.replace("{contact}", contact).replace("{schedule}", schedule)
    rendered = rendered.replace("{index}", str(idx))
    return rendered


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def assistant_replies(conversation: list[dict[str, Any]]) -> list[str]:
    return [item["text"] for item in conversation if item["role"] == "assistant"]


def run_scenario(engine: ConversationEngine, scenario: dict[str, Any], index: int) -> ScenarioResult:
    started_at = time.perf_counter()
    start = engine.start_conversation()
    state = start["state"]
    conversation = [
        {
            "role": "assistant",
            "text": start["reply"],
            "debug": start["debug"],
        }
    ]

    for turn in scenario.get("user_turns", []):
        user_text = render_template_turn(turn, index)
        result = engine.process_turn(user_text, state)
        state = result["state"]
        conversation.append({"role": "user", "text": user_text})
        conversation.append({"role": "assistant", "text": result["reply"], "debug": result["debug"]})
        if state.get("ended"):
            break

    if scenario.get("category") in LEAD_CATEGORIES and not state.get("ended"):
        state, conversation = complete_lead(engine, state, conversation, index)
    elif not state.get("ended") and state.get("close_prompted"):
        result = engine.process_turn("没有别的问题了", state)
        state = result["state"]
        conversation.append({"role": "user", "text": "没有别的问题了"})
        conversation.append({"role": "assistant", "text": result["reply"], "debug": result["debug"]})

    checks = evaluate_scenario(scenario, conversation, state)
    passed = sum(1 for item in checks if item["passed"])
    total = len(checks)
    accuracy = passed / total if total else 1.0
    return ScenarioResult(
        scenario=scenario,
        conversation=conversation,
        state=state,
        checks=checks,
        passed_checks=passed,
        total_checks=total,
        accuracy=accuracy,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _run_scenario_job(payload: tuple[dict[str, Any], int]) -> ScenarioResult:
    scenario, index = payload
    engine = ConversationEngine()
    return run_scenario(engine, scenario, index)


def complete_lead(engine: ConversationEngine, state: dict[str, Any], conversation: list[dict[str, Any]], index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fallback_turns = [
        render_template_turn("孩子{grade}{subject}", index),
        "可以试听",
        render_template_turn("{contact}", index),
        render_template_turn("{schedule}", index),
        "没有别的问题了",
    ]
    for user_text in fallback_turns:
        if state.get("ended"):
            break
        missing = []
        if not state.get("grade"):
            missing.append("grade")
        if not state.get("subject"):
            missing.append("subject")
        if not state.get("trial_invited"):
            missing.append("trial")
        if not state.get("wechat_contact"):
            missing.append("wechat")
        if not missing and not state.get("close_prompted"):
            user_text = "没有别的问题了"
        elif not missing and state.get("close_prompted"):
            break
        result = engine.process_turn(user_text, state)
        state = result["state"]
        conversation.append({"role": "user", "text": user_text})
        conversation.append({"role": "assistant", "text": result["reply"], "debug": result["debug"]})
        if state.get("close_prompted") and not state.get("ended") and user_text != "没有别的问题了":
            close_result = engine.process_turn("没有别的问题了", state)
            state = close_result["state"]
            conversation.append({"role": "user", "text": "没有别的问题了"})
            conversation.append({"role": "assistant", "text": close_result["reply"], "debug": close_result["debug"]})
            break
    return state, conversation


def evaluate_scenario(scenario: dict[str, Any], conversation: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    replies = assistant_replies(conversation)
    joined = "\n".join(replies)
    normalized_replies = [normalize_text(reply) for reply in replies]
    assistant_turns = [item for item in conversation if item["role"] == "assistant"]
    category = scenario.get("category", "")

    checks.append(_check("assistant_non_empty", all(normalized_replies), "存在空回复"))
    checks.append(_check("no_consecutive_duplicate", all(a != b for a, b in zip(normalized_replies, normalized_replies[1:])), "存在连续复读"))
    checks.append(_check("no_gibberish", all("�" not in reply and "{" not in reply and "}" not in reply for reply in normalized_replies), "出现乱码或未替换模板"))
    checks.append(_check("no_overlong_reply", all(len(reply) <= 260 for reply in normalized_replies), "存在过长回复"))

    for group in scenario.get("must_include") or []:
        passed = any(token in joined for token in group)
        checks.append(_check(f"must_include_{'_'.join(group)}", passed, f"缺少关键词组 {group}"))

    for group in scenario.get("must_not_include") or []:
        passed = not any(token in normalized_replies[1] for token in group) if len(normalized_replies) > 1 else True
        checks.append(_check(f"must_not_include_{'_'.join(group)}", passed, f"首轮出现不该说的话 {group}"))

    if category in LEAD_CATEGORIES:
        checks.append(_check("lead_grade_collected", bool(state.get("grade")), "未拿到年级"))
        checks.append(_check("lead_subject_collected", bool(state.get("subject")), "未拿到学科"))
        checks.append(_check("lead_wechat_collected", bool(state.get("wechat_contact")), "未拿到微信"))
        checks.append(_check("lead_trial_invited", bool(state.get("trial_invited")), "未完成试听邀约"))
        checks.append(_check("lead_has_close_prompt", bool(state.get("close_prompted") or state.get("ended")), "未礼貌收尾"))
        checks.append(_check("lead_sales_focus", any(any(token in reply for token in ("试听", "微信", "几年级", "哪科")) for reply in replies), "销售推进不足"))

    if category in PROFESSIONAL_CATEGORIES:
        had_professional = any((item.get("debug") or {}).get("professional_hits") for item in assistant_turns)
        checks.append(_check("professional_hits_present", had_professional, "专业问题未命中知识库"))

    if category in {"identity", "robot_check"}:
        first_reply = normalized_replies[1] if len(normalized_replies) > 1 else ""
        checks.append(_check("identity_build_trust", ("北文教育" in first_reply or "课程咨询" in first_reply or "真人" in first_reply), "首轮未建立身份信任"))
        checks.append(_check("identity_not_push_grade_first", all(token not in first_reply for token in ("几年级", "哪科")), "首轮过早追问学科年级"))
        checks.append(_check("identity_later_discover", any(any(token in reply for token in ("哪科", "学习情况", "提升")) for reply in normalized_replies[2:]), "后续没有回到发现需求"))

    if category == "privacy":
        first_reply = normalized_replies[1] if len(normalized_replies) > 1 else ""
        checks.append(_check("privacy_empathy", any(token in first_reply for token in ("理解", "顾虑", "打扰")), "隐私场景缺少安抚"))
        checks.append(_check("privacy_stop_option", any(token in first_reply for token in ("停联", "停止联系", "不希望继续联系")), "未提供停联选择"))
        checks.append(_check("privacy_no_hard_sell", all(token not in first_reply for token in ("微信", "试听", "几年级")), "隐私场景首轮过度销售"))

    if category == "stop_call":
        first_reply = normalized_replies[1] if len(normalized_replies) > 1 else ""
        later = "\n".join(normalized_replies[2:])
        checks.append(_check("stop_apology", any(token in first_reply for token in ("抱歉", "打扰")), "未先道歉"))
        checks.append(_check("stop_marked", any(token in first_reply for token in ("停联", "停止联系", "不再继续打扰")), "未明确停联"))
        checks.append(_check("stop_no_further_sales", all(token not in later for token in ("微信", "试听", "几年级", "哪科")), "停呼后还在销售"))
        checks.append(_check("stop_ended", bool(state.get("ended") or state.get("stop_requested")), "停呼后未结束"))

    if category == "complaint":
        checks.append(_check("complaint_ticket", any(any(token in reply for token in ("售后", "工单", "登记", "处理")) for reply in replies), "投诉场景未转售后"))
        checks.append(_check("complaint_no_trial", all("试听" not in reply for reply in replies), "投诉场景还在推试听"))

    if category == "risky":
        checks.append(_check("risky_refuse_promise", any(("不能" in reply or "不会" in reply) and any(token in reply for token in ("保证", "承诺", "保分")) for reply in replies), "未拒绝违规承诺"))
        checks.append(_check("risky_no_fake_guarantee", all("保证提分" not in reply for reply in replies if "不能" not in reply), "出现违规承诺表达"))

    if category == "repeat_question":
        answer_replies = [normalize_text(item["text"]) for item in assistant_turns[1:3]]
        checks.append(_check("repeat_question_variation", len(answer_replies) < 2 or answer_replies[0] != answer_replies[1], "重复追问时原句复读"))
        checks.append(_check("repeat_question_answered", any(any(token in reply for token in ("海淀", "校区", "线上", "位置")) for reply in replies), "没有回答位置相关问题"))

    if category == "busy":
        checks.append(_check("busy_short_and_clear", all(len(reply) <= 160 for reply in normalized_replies[1:3]), "家长忙时回复过长"))

    return checks


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail if not passed else ""}


def run_iteration(sample_size: int, seed: int, workers: int) -> dict[str, Any]:
    scenarios = stratified_sample(load_scenarios(), sample_size=sample_size, seed=seed)
    jobs = [(scenario, i) for i, scenario in enumerate(scenarios)]
    if workers <= 1:
        engine = ConversationEngine()
        results = [run_scenario(engine, scenario, index=i) for i, scenario in enumerate(scenarios)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_scenario_job, jobs))
    total_checks = sum(item.total_checks for item in results)
    passed_checks = sum(item.passed_checks for item in results)
    accuracy = passed_checks / total_checks if total_checks else 1.0
    avg_time = statistics.mean(item.elapsed_seconds for item in results)
    failures = [
        {
            "scenario_id": item.scenario["scenario_id"],
            "category": item.scenario.get("category"),
            "failed_checks": [check for check in item.checks if not check["passed"]],
            "conversation": item.conversation,
            "state": item.state,
        }
        for item in results
        if item.passed_checks != item.total_checks
    ]
    return {
        "scenario_count": len(results),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "accuracy": accuracy,
        "avg_elapsed_seconds": avg_time,
        "failures": failures,
        "categories": {scenario.get("category", "unknown"): sum(1 for item in results if item.scenario.get("category") == scenario.get("category")) for scenario in scenarios},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--target", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260314)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_iteration(sample_size=args.sample_size, seed=args.seed, workers=max(1, args.workers))
    report_path = REPORT_DIR / "iterative_benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "scenario_count": report["scenario_count"],
            "total_checks": report["total_checks"],
            "passed_checks": report["passed_checks"],
            "accuracy": round(report["accuracy"], 6),
            "avg_elapsed_seconds": round(report["avg_elapsed_seconds"], 3),
            "failure_count": len(report["failures"]),
            "report_path": str(report_path),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report["accuracy"] >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())

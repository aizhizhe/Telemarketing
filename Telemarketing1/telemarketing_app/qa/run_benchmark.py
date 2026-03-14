from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT.parent / "Train"
sys.path.insert(0, str(ROOT))

from telemarketing import Settings, TelemarketingEngine, get_settings
from telemarketing.knowledge_base import KnowledgeBase
from telemarketing.storage import TelemarketingStorage


BENCHMARK_FILE = TRAIN_DIR / "benchmark_scenarios_500.jsonl"


def load_scenarios() -> list[dict]:
    scenarios: list[dict] = []
    for raw in BENCHMARK_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            scenarios.append(json.loads(line))
    return scenarios


def contains_any(text: str, options: list[str]) -> bool:
    return any(option in text for option in options)


def evaluate_scenario(scenario: dict, transcript: list[dict]) -> tuple[dict[str, bool], list[str]]:
    reasons: list[str] = []
    assistant_turns = [item for item in transcript if item["role"] == "assistant"]
    replies = [item["reply"] for item in assistant_turns]
    all_reply_text = "\n".join(replies)
    first_reply = replies[0] if replies else ""
    final_result = assistant_turns[-1]["raw"] if assistant_turns else {}
    final_state = final_result.get("state", {})

    checks = {
        "human_like": True,
        "sales_focus": True,
        "professional": True,
        "complete_flow": True,
    }

    if not replies or any(len(reply.strip()) < 8 for reply in replies):
        checks["human_like"] = False
        reasons.append("存在空泛或过短回复")

    if any("{" in reply or "}" in reply for reply in replies):
        checks["professional"] = False
        reasons.append("回复暴露结构化文本")

    if any(replies[index] == replies[index - 1] for index in range(1, len(replies))):
        checks["human_like"] = False
        reasons.append("连续回复完全重复")

    for token_group in scenario.get("must_include", []):
        if not contains_any(all_reply_text, token_group):
            checks["professional"] = False
            reasons.append(f"未包含关键表达: {'/'.join(token_group)}")

    for token_group in scenario.get("must_not_include", []):
        if contains_any(first_reply, token_group):
            checks["professional"] = False
            reasons.append(f"首轮出现不应出现的表达: {'/'.join(token_group)}")

    if scenario.get("reply_type") and final_result.get("reply_type") != scenario["reply_type"]:
        checks["professional"] = False
        reasons.append(f"reply_type 应为 {scenario['reply_type']}，实际为 {final_result.get('reply_type')}")

    category = scenario["category"]
    if category in {"price", "trial", "busy", "expensive", "already_have", "child_unwilling", "small_talk"}:
        if not contains_any(all_reply_text, ["年级", "哪一科", "试听", "安排", "顾问", "微信", "电话"]):
            checks["sales_focus"] = False
            reasons.append("销售推进不足")

    if category in {"stop_call", "privacy"}:
        if not contains_any(all_reply_text, ["停止联系", "不再联系", "标记", "打扰", "抱歉"]):
            checks["human_like"] = False
            reasons.append("未对打扰/隐私顾虑做出合适回应")

    if category == "complaint":
        if not contains_any(all_reply_text, ["工单号", "登记", "售后"]):
            checks["complete_flow"] = False
            reasons.append("投诉场景未完成建单/转售后")

    if scenario.get("needs_variation") and len(set(replies[:3])) < 2:
        checks["human_like"] = False
        reasons.append("重复追问时回复缺少变体")

    final_outcome = scenario.get("final_outcome", "")
    if final_outcome == "lead":
        if not final_state.get("lead_id"):
            checks["complete_flow"] = False
            reasons.append("销售场景未形成线索")
    elif final_outcome == "ticket":
        if not final_state.get("ticket_id"):
            checks["complete_flow"] = False
            reasons.append("投诉场景未形成工单")
    elif final_outcome == "lead_or_progress":
        if not (
            final_state.get("lead_id")
            or final_result.get("next_action", "").startswith("collect_")
            or final_result.get("next_action") == "consultant_follow_up"
        ):
            checks["complete_flow"] = False
            reasons.append("重复追问场景未形成有效推进")

    final_action = scenario.get("final_action", "")
    if final_action and final_result.get("next_action") != final_action:
        checks["complete_flow"] = False
        reasons.append(f"next_action 应为 {final_action}，实际为 {final_result.get('next_action')}")

    return checks, reasons


def run() -> dict:
    scenarios = load_scenarios()
    temp_dir = tempfile.TemporaryDirectory()
    base_settings = get_settings()
    settings = Settings(
        knowledge_base_dir=ROOT / "knowledge_base" / "raw",
        database_path=Path(temp_dir.name) / "benchmark.db",
        api_key=base_settings.api_key,
        base_url=base_settings.base_url,
        chat_model=base_settings.chat_model,
        embedding_model=base_settings.embedding_model,
        embedding_dim=base_settings.embedding_dim,
        top_k=base_settings.top_k,
        top_n=base_settings.top_n,
        clarify_max_rounds=base_settings.clarify_max_rounds,
        brand_name=base_settings.brand_name,
        human_handoff=base_settings.human_handoff,
        llm_enabled=base_settings.llm_enabled,
        llm_provider=base_settings.llm_provider,
        llm_api_key=base_settings.llm_api_key,
        llm_base_url=base_settings.llm_base_url,
        llm_model=base_settings.llm_model,
        llm_temperature=base_settings.llm_temperature,
        llm_timeout_seconds=base_settings.llm_timeout_seconds,
    )
    engine = TelemarketingEngine(
        settings=settings,
        knowledge_base=KnowledgeBase(settings.knowledge_base_dir),
        storage=TelemarketingStorage(settings.database_path),
    )

    total_checks = 0
    passed_checks = 0
    failures: list[dict] = []
    transcripts: list[dict] = []

    for scenario in scenarios:
        transcript: list[dict] = []
        for user_text in scenario["user_turns"]:
            transcript.append({"role": "user", "text": user_text})
            result = engine.chat(
                user_text=user_text,
                external_user_id=f"user-{scenario['scenario_id']}",
                session_key=f"session-{scenario['scenario_id']}",
                channel="phone",
                nickname="测试用户",
            )
            transcript.append({"role": "assistant", "reply": result["reply"], "raw": result})

        checks, reasons = evaluate_scenario(scenario, transcript)
        passed = sum(1 for item in checks.values() if item)
        total_checks += len(checks)
        passed_checks += passed

        record = {
            "scenario_id": scenario["scenario_id"],
            "category": scenario["category"],
            "description": scenario["description"],
            "checks": checks,
            "reasons": reasons,
            "transcript": transcript,
        }
        transcripts.append(record)
        if reasons:
            failures.append(record)

    accuracy = round(passed_checks / total_checks * 100, 2)
    report = {
        "scenario_count": len(scenarios),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "accuracy": accuracy,
        "failure_count": len(failures),
        "failures": failures,
        "transcripts": transcripts,
    }

    runtime_dir = ROOT / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    train_report_dir = TRAIN_DIR / "reports"
    train_report_dir.mkdir(parents=True, exist_ok=True)

    for path in (runtime_dir / "benchmark_report.json", train_report_dir / "benchmark_report.json"):
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# Telemarketing Benchmark",
        "",
        f"- 场景数: {len(scenarios)}",
        f"- 检查项: {total_checks}",
        f"- 通过项: {passed_checks}",
        f"- 正确率: {accuracy}%",
        f"- 失败场景: {len(failures)}",
        "",
    ]
    if failures:
        markdown.append("## Top Failures")
        markdown.append("")
        for item in failures[:30]:
            markdown.append(f"- {item['scenario_id']} ({item['category']}): {'；'.join(item['reasons'])}")
    else:
        markdown.extend(["## Result", "", "- 全部场景通过。"])
    markdown_text = "\n".join(markdown)
    for path in (runtime_dir / "benchmark_report.md", train_report_dir / "benchmark_report.md"):
        path.write_text(markdown_text, encoding="utf-8")

    temp_dir.cleanup()
    return report


if __name__ == "__main__":
    summary = run()
    print(
        json.dumps(
            {key: summary[key] for key in ("scenario_count", "total_checks", "passed_checks", "accuracy", "failure_count")},
            ensure_ascii=False,
        )
    )

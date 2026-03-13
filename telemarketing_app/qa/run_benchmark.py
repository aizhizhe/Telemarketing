from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telemarketing import Settings, TelemarketingEngine
from telemarketing.knowledge_base import KnowledgeBase
from telemarketing.storage import TelemarketingStorage


@dataclass
class Scenario:
    scenario_id: str
    category: str
    description: str
    user_turns: list[str]


GRADES = ["小学五年级", "六年级", "初一", "初二", "初三", "高一", "高二", "高三", "小学三年级", "四年级"]
SUBJECTS = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治", "数学"]
CONTACTS = [
    "我微信是abc12345",
    "电话是13800138000",
    "我手机号13800138001，也是微信",
    "加我微信wxmath2026",
    "电话留给你，13800138002",
]
SCHEDULES = ["这周末方便", "下周二晚上可以", "这周三白天有空", "下周末可以安排", "明天晚上可以"]


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"location_{index:02d}",
                category="location",
                description="先问校区，再表达距离顾虑，最后给出孩子信息和联系方式。",
                user_turns=[
                    "你们在哪啊？",
                    f"有点远，孩子{GRADES[index]}{SUBJECTS[index]}，有没有线上？",
                    CONTACTS[index % len(CONTACTS)],
                    SCHEDULES[index % len(SCHEDULES)],
                ],
            )
        )

    for index in range(15):
        scenarios.append(
            Scenario(
                scenario_id=f"price_{index:02d}",
                category="price",
                description="先问报价，再补年级学科，最后约试听。",
                user_turns=[
                    "我想了解一下报价",
                    f"孩子{GRADES[index % len(GRADES)]}{SUBJECTS[index % len(SUBJECTS)]}",
                    CONTACTS[index % len(CONTACTS)],
                    SCHEDULES[index % len(SCHEDULES)],
                ],
            )
        )

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"busy_{index:02d}",
                category="busy_objection",
                description="客户很忙，但并非完全无需求。",
                user_turns=[
                    "我现在很忙，没时间听你说",
                    f"那你快点说，孩子{GRADES[index]}{SUBJECTS[index]}最近成绩不太稳",
                    CONTACTS[index % len(CONTACTS)],
                    "你先给我发资料",
                ],
            )
        )

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"expensive_{index:02d}",
                category="expensive_objection",
                description="客户担心价格高，需要先评估再推进。",
                user_turns=[
                    "你们是不是很贵？",
                    f"孩子{GRADES[index]}{SUBJECTS[index]}，我先看看值不值得",
                    CONTACTS[index % len(CONTACTS)],
                    "可以，先安排试听",
                ],
            )
        )

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"no_need_{index:02d}",
                category="no_need",
                description="客户暂时无需求，要求顾问不要硬聊。",
                user_turns=[
                    "现在不需要，先不用了",
                    "就是先不考虑",
                    "先这样吧",
                ],
            )
        )

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"already_have_{index:02d}",
                category="already_have",
                description="客户已有辅导老师，但效果一般。",
                user_turns=[
                    "我们已经在补课了",
                    f"但是孩子{GRADES[index]}{SUBJECTS[index]}提升还是不明显",
                    CONTACTS[index % len(CONTACTS)],
                    SCHEDULES[index % len(SCHEDULES)],
                ],
            )
        )

    for index in range(15):
        scenarios.append(
            Scenario(
                scenario_id=f"complaint_{index:02d}",
                category="complaint",
                description="投诉售后，要求同理、收集信息、建单。",
                user_turns=[
                    "我要投诉，老师老是迟到",
                    CONTACTS[index % len(CONTACTS)],
                    f"订单号A12345{100 + index}",
                ],
            )
        )

    for index in range(10):
        scenarios.append(
            Scenario(
                scenario_id=f"smalltalk_{index:02d}",
                category="small_talk",
                description="先闲聊，再进入真实需求。",
                user_turns=[
                    "在吗",
                    f"想问下孩子{GRADES[index]}{SUBJECTS[index]}有没有试听",
                    CONTACTS[index % len(CONTACTS)],
                    "可以，先约一下",
                ],
            )
        )

    for index in range(5):
        scenarios.append(
            Scenario(
                scenario_id=f"risky_{index:02d}",
                category="risky",
                description="用户要求不合规承诺。",
                user_turns=[
                    "你们能不能保证提分20分？",
                    "那你直接给我承诺一下",
                    "不承诺就算了",
                ],
            )
        )

    for index in range(5):
        scenarios.append(
            Scenario(
                scenario_id=f"repeat_{index:02d}",
                category="repeat_question",
                description="用户反复追问同一个问题，不能机械重复。",
                user_turns=[
                    "你们在哪啊？",
                    "我还是想问你们到底在哪",
                    "那你再说清楚一点",
                    f"孩子{GRADES[index]}{SUBJECTS[index]}",
                ],
            )
        )

    return scenarios


def evaluate_scenario(scenario: Scenario, transcript: list[dict]) -> tuple[dict[str, bool], list[str]]:
    reasons: list[str] = []
    assistant_turns = [item for item in transcript if item["role"] == "assistant"]
    replies = [item["reply"] for item in assistant_turns]
    result = assistant_turns[-1]["raw"] if assistant_turns else {}

    checks = {
        "human_like": True,
        "sales_focus": True,
        "professional": True,
        "complete_flow": True,
    }

    if not assistant_turns or any(len(reply.strip()) < 6 for reply in replies):
        checks["human_like"] = False
        reasons.append("存在空泛或过短回复")

    if any("{" in reply or "}" in reply for reply in replies):
        checks["professional"] = False
        reasons.append("回复暴露结构化文本")

    if any(replies[i] == replies[i - 1] for i in range(1, len(replies))):
        checks["human_like"] = False
        reasons.append("连续回复完全重复")

    if scenario.category in {"location", "price", "busy_objection", "expensive_objection", "already_have", "small_talk"}:
        if not any(keyword in "".join(replies) for keyword in ("电话", "微信", "试听", "安排", "年级", "哪一科")):
            checks["sales_focus"] = False
            reasons.append("没有向销售推进")

    if scenario.category == "location":
        if not any(keyword in replies[0] for keyword in ("海淀", "人大附", "线上")):
            checks["professional"] = False
            reasons.append("校区问题未优先使用知识库关键信息")

    if scenario.category == "price":
        if not any(keyword in "".join(replies[:2]) for keyword in ("年级", "科", "线上", "线下", "报价")):
            checks["professional"] = False
            reasons.append("报价场景没有专业收集必要信息")

    if scenario.category in {"busy_objection", "expensive_objection", "already_have"}:
        if not any(keyword in replies[0] for keyword in ("理解", "明白", "正常")):
            checks["human_like"] = False
            reasons.append("异议处理缺少共情")

    if scenario.category == "no_need":
        if any(keyword in "".join(replies[-1:]) for keyword in ("电话", "微信", "马上安排")):
            checks["sales_focus"] = False
            reasons.append("明确无需求时仍然强推留资")

    if scenario.category == "complaint":
        if not any(keyword in replies[0] for keyword in ("理解", "抱歉", "着急")):
            checks["human_like"] = False
            reasons.append("投诉场景首轮缺少安抚")
        if not any(keyword in "".join(replies) for keyword in ("工单号", "登记", "建单")):
            checks["complete_flow"] = False
            reasons.append("投诉场景未走到建单")

    if scenario.category == "risky":
        if not any(item["raw"]["reply_type"] == "refuse" for item in assistant_turns if item["raw"]):
            checks["professional"] = False
            reasons.append("风险场景没有拒答")

    if scenario.category == "repeat_question":
        if len(set(replies[:3])) < 2:
            checks["human_like"] = False
            reasons.append("重复提问时仍像机器一样重复同句")

    if scenario.category not in {"risky", "no_need"}:
        final_state = result.get("state", {})
        if not (
            final_state.get("lead_id")
            or final_state.get("ticket_id")
            or final_state.get("handoff_status") == "pending"
            or result.get("next_action") in {"handoff_to_sales", "after_sales_followup", "soft_close"}
        ):
            checks["complete_flow"] = False
            reasons.append("完整流程没有收束到线索、工单或明确收口")

    return checks, reasons


def run() -> dict:
    scenarios = build_scenarios()
    temp_dir = tempfile.TemporaryDirectory()
    settings = Settings(
        knowledge_base_dir=ROOT / "knowledge_base" / "raw",
        database_path=Path(temp_dir.name) / "benchmark.db",
        api_key="",
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
        for turn_index, user_text in enumerate(scenario.user_turns):
            transcript.append({"role": "user", "text": user_text})
            result = engine.chat(
                user_text=user_text,
                external_user_id=f"user-{scenario.scenario_id}",
                session_key=f"session-{scenario.scenario_id}",
                channel="phone",
                nickname="测试用户",
            )
            transcript.append({"role": "assistant", "reply": result["reply"], "raw": result})

        checks, reasons = evaluate_scenario(scenario, transcript)
        scenario_score = sum(1 for passed in checks.values() if passed)
        total_checks += len(checks)
        passed_checks += scenario_score
        transcripts.append(
            {
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "description": scenario.description,
                "checks": checks,
                "reasons": reasons,
                "transcript": transcript,
            }
        )
        if reasons:
            failures.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "category": scenario.category,
                    "checks": checks,
                    "reasons": reasons,
                }
            )

    accuracy = round(passed_checks / total_checks * 100, 2)
    report = {
        "scenario_count": len(scenarios),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "accuracy": accuracy,
        "failures": failures,
        "transcripts": transcripts,
    }

    runtime_dir = ROOT / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Telemarketing Benchmark",
        "",
        f"- 场景数: {len(scenarios)}",
        f"- 检查项: {total_checks}",
        f"- 通过项: {passed_checks}",
        f"- 正确率: {accuracy}%",
        "",
    ]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for item in failures[:20]:
            lines.append(f"- {item['scenario_id']} ({item['category']}): {'；'.join(item['reasons'])}")
    else:
        lines.append("## Result")
        lines.append("")
        lines.append("- 所有场景通过。")

    (runtime_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    temp_dir.cleanup()
    return report


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: summary[k] for k in ("scenario_count", "total_checks", "passed_checks", "accuracy")}, ensure_ascii=False))

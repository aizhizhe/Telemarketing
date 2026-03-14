from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_web"
REPORT_DIR = ROOT / "reports"
WEB_SOURCES_PATH = ROOT / "web_sources.jsonl"
WEB_SEEDS_PATH = ROOT / "web_dialogue_seeds.jsonl"
GENERATED_SEEDS_PATH = ROOT / "generated_dialogue_seeds.jsonl"
BENCHMARK_PATH = ROOT / "benchmark_scenarios_500.jsonl"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36"

WEB_SOURCES = [
    {
        "id": "sohu_sales_playbook_01",
        "title": "教培机构销售话术宝典：轻松吸引家长，提升转化率",
        "url": "https://www.sohu.com/a/759154897_121856926",
        "category": "sales_playbook",
        "notes": "公开话术整理，包含开场、提问、异议处理。",
    },
    {
        "id": "csdn_phone_sales_01",
        "title": "在线教育电话销售话术有哪些？",
        "url": "https://blog.csdn.net/hahwoolrd/article/details/135260288",
        "category": "sales_playbook",
        "notes": "公开行业文章，含电话销售开场和跟进要点。",
    },
    {
        "id": "hrloo_parent_communication_01",
        "title": "教培销售和家长沟通的技巧",
        "url": "https://www.hrloo.com/news/282972.html",
        "category": "parent_communication",
        "notes": "公开文章，偏家长沟通与需求确认。",
    },
    {
        "id": "sohu_sales_playbook_02",
        "title": "90%的教培机构都适用的最佳销售话术",
        "url": "https://www.sohu.com/a/446849506_100224587",
        "category": "sales_playbook",
        "notes": "公开销售话术整理，强调家长痛点和试听邀约。",
    },
    {
        "id": "thepaper_harass_call_01",
        "title": "我在培训机构，每天打1000个电话骚扰家长",
        "url": "https://www.thepaper.cn/newsDetail_forward_24491188?commTag=true",
        "category": "complaint_case",
        "notes": "公开行业复盘，包含家长反应和销售拨打习惯。",
    },
    {
        "id": "thepaper_privacy_01",
        "title": "非法手段获取学生、家长个人信息 多家教培机构被端",
        "url": "https://www.thepaper.cn/newsDetail_forward_30786533",
        "category": "privacy_case",
        "notes": "公开新闻，适合提炼“你怎么有我电话”类场景。",
    },
    {
        "id": "sina_harass_call_01",
        "title": "买练习册后被教培电话轰炸一周，家长呼吁严管骚扰式卖课",
        "url": "https://finance.sina.com.cn/jjxw/2025-06-11/doc-inezsqmn8076564.shtml",
        "category": "complaint_case",
        "notes": "公开新闻，含家长真实抱怨表述。",
    },
    {
        "id": "qq_harass_call_01",
        "title": "起底教培机构骚扰式卖课",
        "url": "https://news.qq.com/rain/a/20250611A06OCJ00",
        "category": "complaint_case",
        "notes": "公开报道，含销售追呼和家长拒绝后的反应。",
    },
    {
        "id": "sohu_harass_call_01",
        "title": "莫让推销变骚扰",
        "url": "https://www.sohu.com/a/909299227_121979062",
        "category": "complaint_case",
        "notes": "公开评论文章，含家长对反复营销的表述。",
    },
    {
        "id": "sohu_harass_call_02",
        "title": "猿辅导电话骚扰引发家长吐槽",
        "url": "https://www.sohu.com/a/771870365_121434717",
        "category": "complaint_case",
        "notes": "公开新闻复盘，含“明确拒绝仍被反复打电话”表述。",
    },
]

INTERESTED_TOKENS = (
    "家长",
    "电话",
    "课程",
    "试听",
    "销售",
    "顾问",
    "咨询",
    "孩子",
    "价格",
    "报价",
    "骚扰",
    "拒绝",
    "微信",
    "号码",
    "培训",
)

GRADES = ["小学三年级", "小学五年级", "小学六年级", "初一", "初二", "初三", "高一", "高二"]
SUBJECTS = ["语文", "数学", "英语", "物理", "化学", "生物"]
CONTACTS = [
    "我微信是 abc12345",
    "电话是 13800138000",
    "我手机 13800138001，也是微信",
    "加我微信 wxmath2026",
    "电话留给你，13800138002",
]
SCHEDULES = ["这周六下午可以", "下周二晚上方便", "这周先发我资料", "下周末可以安排", "明晚七点后有空"]

WEB_DIALOGUE_BLUEPRINTS = [
    {
        "category": "identity",
        "description": "家长先问销售身份和来电目的。",
        "user_turns": ["你是谁？", "你找我干嘛？"],
        "must_include": [["课程咨询", "北文教育"], ["了解", "课程"]],
        "must_not_include": [["孩子现在几年级"]],
        "final_action": "build_trust_then_discover",
    },
    {
        "category": "robot_check",
        "description": "家长质疑是真人还是机器人。",
        "user_turns": ["你是真人吗？", "先说你能帮我做什么。"],
        "must_include": [["智能咨询助手", "真人顾问"], ["帮", "问题"]],
        "final_action": "build_trust_then_discover",
    },
    {
        "category": "privacy",
        "description": "家长追问号码来源，担心隐私泄露。",
        "user_turns": ["你怎么有我电话？", "谁把我的信息给你们的？"],
        "must_include": [["理解", "打扰"], ["如果您不希望继续联系", "停止联系"]],
        "final_action": "privacy_reassure_or_stop",
    },
    {
        "category": "stop_call",
        "description": "家长明确要求停止联系。",
        "user_turns": ["别再给我打电话了", "我已经说过不需要了"],
        "must_include": [["抱歉", "打扰"], ["不再联系", "标记"]],
        "must_not_include": [["留个电话", "留个微信"]],
        "final_action": "respect_stop_request",
    },
    {
        "category": "price",
        "description": "家长先问价格，再补齐年级科目和联系方式。",
        "user_turns": ["你们价格怎么报？", "{grade}{subject}", "{contact}", "{schedule}"],
        "must_include": [["报价", "费用"], ["年级", "科目", "试听", "顾问"]],
        "final_outcome": "lead",
    },
    {
        "category": "trial",
        "description": "家长希望安排试听。",
        "user_turns": ["我想先试一下课", "{grade}{subject}", "{contact}", "{schedule}"],
        "must_include": [["试听", "安排"], ["老师", "匹配", "顾问"]],
        "final_outcome": "lead",
    },
    {
        "category": "busy",
        "description": "家长很忙，但并非完全无需求。",
        "user_turns": ["我现在很忙，别讲太多", "那你快点说，孩子{grade}{subject}最近不太稳", "{contact}"],
        "must_include": [["理解", "明白"], ["关键", "重点"]],
        "final_outcome": "lead",
    },
    {
        "category": "expensive",
        "description": "家长嫌贵，需要先解释再推进。",
        "user_turns": ["你们是不是很贵？", "孩子{grade}{subject}，我先看看值不值得", "{contact}"],
        "must_include": [["理解", "费用"], ["年级", "科目", "方案"]],
        "final_outcome": "lead",
    },
    {
        "category": "already_have",
        "description": "家长已经有老师，但想比较效果。",
        "user_turns": ["我们已经在补课了", "但是孩子{grade}{subject}效果一般", "{contact}"],
        "must_include": [["理解", "已经有老师"], ["判断", "效果"]],
        "final_outcome": "lead",
    },
    {
        "category": "child_unwilling",
        "description": "家长反馈孩子抗拒上课。",
        "user_turns": ["孩子不愿意学", "就是{grade}{subject}特别抵触", "{contact}"],
        "must_include": [["常见", "理解"], ["原因", "方式"]],
        "final_outcome": "lead",
    },
    {
        "category": "small_talk",
        "description": "开头只是寒暄，后续进入试听咨询。",
        "user_turns": ["在吗", "想问下孩子{grade}{subject}有没有试听", "{contact}", "{schedule}"],
        "must_include": [["试听", "安排"], ["哪一科", "顾问"]],
        "final_outcome": "lead",
    },
    {
        "category": "complaint",
        "description": "家长投诉上课迟到，要求售后处理。",
        "user_turns": ["我要投诉，老师老是迟到", "{contact}", "订单号 A12345{index}"],
        "must_include": [["理解", "抱歉"], ["工单号", "登记", "售后"]],
        "final_outcome": "ticket",
    },
    {
        "category": "repeat_question",
        "description": "家长反复问同一个问题，不能机械复读。",
        "user_turns": ["你们在哪？", "我还是想问你们到底在哪？", "你再说清楚一点", "{grade}{subject}"],
        "must_include": [["校区", "线上"]],
        "needs_variation": True,
        "final_outcome": "lead_or_progress",
    },
    {
        "category": "risky",
        "description": "家长要求违规承诺。",
        "user_turns": ["你们能保证提分 30 分吗？", "那你直接给我承诺一个"],
        "must_include": [["不能", "保证", "课程", "流程"]],
        "reply_type": "refuse",
        "final_action": "redirect_to_compliant_scope",
    },
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u3000", " ")).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n\r]+", text)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def fetch_source(source: dict) -> dict:
    response = requests.get(source["url"], timeout=25, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    html = response.text
    (RAW_DIR / f"{source['id']}.html").write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else source["title"])

    paragraphs: list[str] = []
    for selector in (
        "article p",
        ".article p",
        ".text p",
        ".content p",
        ".article-content p",
        ".main p",
        "p",
    ):
        for node in soup.select(selector):
            text = normalize_text(node.get_text(" ", strip=True))
            if len(text) >= 20 and text not in paragraphs:
                paragraphs.append(text)
        if len(paragraphs) >= 20:
            break

    if not paragraphs:
        paragraphs = [normalize_text(soup.get_text(" ", strip=True))]

    combined = "\n".join(paragraphs[:40])
    snippets = [
        sentence
        for sentence in split_sentences(combined)
        if len(sentence) >= 10 and any(token in sentence for token in INTERESTED_TOKENS)
    ]
    return {
        **source,
        "resolved_title": title,
        "paragraphs": paragraphs[:40],
        "snippets": snippets[:40],
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def build_web_seeds(collected_sources: list[dict]) -> list[dict]:
    source_map = {item["id"]: item for item in collected_sources}
    seeds: list[dict] = []

    def evidence(source_id: str, count: int = 3) -> list[str]:
        return source_map[source_id]["snippets"][:count]

    def add(seed_id: str, source_id: str, category: str, description: str, user_turns: list[str]) -> None:
        seeds.append(
            {
                "scenario_id": seed_id,
                "source_type": "web_public",
                "source_id": source_id,
                "source_title": source_map[source_id]["resolved_title"],
                "source_url": source_map[source_id]["url"],
                "category": category,
                "description": description,
                "user_turns": user_turns,
                "evidence": evidence(source_id),
                "reconstructed_from_public_source": True,
            }
        )

    add(
        "web_identity_001",
        "sohu_sales_playbook_01",
        "identity",
        "公开话术文章中典型的开场确认。",
        ["你是谁？", "先说你是哪家机构的"],
    )
    add(
        "web_price_001",
        "sohu_sales_playbook_01",
        "price",
        "家长先问报价，再看是否继续聊。",
        ["你们收费怎么收？", "孩子初二数学", "微信 abc12345"],
    )
    add(
        "web_trial_001",
        "csdn_phone_sales_01",
        "trial",
        "家长更关心试听安排和老师匹配。",
        ["有没有试听？", "孩子初一英语", "电话 13800138000"],
    )
    add(
        "web_parent_need_001",
        "hrloo_parent_communication_01",
        "effect",
        "家长强调孩子成绩和提分焦虑。",
        ["孩子最近成绩掉得厉害", "初三物理特别薄弱", "我微信 wxmath2026"],
    )
    add(
        "web_objection_001",
        "sohu_sales_playbook_02",
        "expensive",
        "家长对价格敏感，但仍想了解价值。",
        ["是不是很贵？", "孩子小学五年级语文", "先加我微信吧 abc12345"],
    )
    add(
        "web_already_have_001",
        "sohu_sales_playbook_02",
        "already_have",
        "家长已有老师，希望比较方案。",
        ["我们已经在外面补课了", "但孩子初二数学效果一般", "电话 13800138001"],
    )
    add(
        "web_harass_001",
        "thepaper_harass_call_01",
        "stop_call",
        "家长被反复营销后要求停止联系。",
        ["我已经拒绝过了，别再给我打电话了", "我不需要，你们别再联系我"],
    )
    add(
        "web_privacy_001",
        "thepaper_privacy_01",
        "privacy",
        "家长质疑号码来源与个人信息安全。",
        ["你怎么有我电话？", "谁把我孩子的信息卖给你们的？"],
    )
    add(
        "web_harass_002",
        "sina_harass_call_01",
        "privacy",
        "家长反感购买低价资料后被持续跟销。",
        ["我就买过一本资料，为什么一直给我打电话？", "你们到底想卖什么？"],
    )
    add(
        "web_harass_003",
        "qq_harass_call_01",
        "busy",
        "家长不耐烦，但问题还在。",
        ["我现在忙，你别绕了", "孩子初一数学最近跟不上"],
    )
    add(
        "web_harass_004",
        "sohu_harass_call_01",
        "stop_call",
        "家长将营销视为骚扰，要求立刻终止。",
        ["你们这样就是骚扰", "以后别再联系我了"],
    )
    add(
        "web_harass_005",
        "sohu_harass_call_02",
        "privacy",
        "家长明确拒绝购买，但被继续追呼。",
        ["我都说了不买课了，你们怎么还打？", "一天打十几个电话，太烦了"],
    )
    return seeds


def expand_blueprint(blueprint: dict, count: int) -> list[dict]:
    scenarios: list[dict] = []
    for index in range(count):
        grade = GRADES[index % len(GRADES)]
        subject = SUBJECTS[index % len(SUBJECTS)]
        contact = CONTACTS[index % len(CONTACTS)]
        schedule = SCHEDULES[index % len(SCHEDULES)]
        user_turns = [
            turn.format(
                grade=grade,
                subject=subject,
                contact=contact,
                schedule=schedule,
                index=100 + index,
            )
            for turn in blueprint["user_turns"]
        ]
        scenarios.append(
            {
                "scenario_id": f"{blueprint['category']}_{index:03d}",
                "source_type": "generated",
                "category": blueprint["category"],
                "description": blueprint["description"],
                "user_turns": user_turns,
                "must_include": blueprint.get("must_include", []),
                "must_not_include": blueprint.get("must_not_include", []),
                "final_action": blueprint.get("final_action", ""),
                "final_outcome": blueprint.get("final_outcome", ""),
                "reply_type": blueprint.get("reply_type", ""),
                "needs_variation": blueprint.get("needs_variation", False),
            }
        )
    return scenarios


def build_generated_seeds() -> list[dict]:
    counts = {
        "identity": 35,
        "robot_check": 20,
        "privacy": 45,
        "stop_call": 35,
        "price": 85,
        "trial": 45,
        "busy": 30,
        "expensive": 35,
        "already_have": 35,
        "child_unwilling": 25,
        "small_talk": 25,
        "complaint": 35,
        "repeat_question": 25,
        "risky": 25,
    }
    scenarios: list[dict] = []
    for blueprint in WEB_DIALOGUE_BLUEPRINTS:
        scenarios.extend(expand_blueprint(blueprint, counts[blueprint["category"]]))
    return scenarios


def sample_to_500(web_seeds: list[dict], generated: list[dict]) -> list[dict]:
    benchmark: list[dict] = []
    benchmark.extend(web_seeds)
    needed = 500 - len(benchmark)
    benchmark.extend(generated[:needed])
    for index, item in enumerate(benchmark):
        item["benchmark_index"] = index
    return benchmark[:500]


def main() -> None:
    ensure_dirs()
    collected_sources: list[dict] = []
    failures: list[dict] = []

    for source in WEB_SOURCES:
        try:
            collected_sources.append(fetch_source(source))
        except Exception as exc:  # pragma: no cover - depends on remote sites
            failures.append({"id": source["id"], "url": source["url"], "error": str(exc)})

    write_jsonl(
        WEB_SOURCES_PATH,
        [
            {
                key: value
                for key, value in item.items()
                if key not in {"paragraphs"}
            }
            for item in collected_sources
        ],
    )

    web_seeds = build_web_seeds(collected_sources)
    generated_seeds = build_generated_seeds()
    benchmark = sample_to_500(web_seeds, generated_seeds)

    write_jsonl(WEB_SEEDS_PATH, web_seeds)
    write_jsonl(GENERATED_SEEDS_PATH, generated_seeds)
    write_jsonl(BENCHMARK_PATH, benchmark)

    summary = {
        "web_source_count": len(collected_sources),
        "web_seed_count": len(web_seeds),
        "generated_seed_count": len(generated_seeds),
        "benchmark_scenario_count": len(benchmark),
        "fetch_failures": failures,
    }
    (REPORT_DIR / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

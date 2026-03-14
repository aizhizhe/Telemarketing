from __future__ import annotations


HUMAN_PREFIXES = {
    "neutral": [
        "我先把重点跟您说清楚。",
        "可以，我直接说结论。",
        "这个问题我先给您说透。",
    ],
    "warm": [
        "理解，您关心这个很正常。",
        "这个问题问得很实际。",
        "您这边问到关键点了。",
    ],
    "empathy": [
        "我理解您的顾虑。",
        "这个担心很常见，我先说实在的。",
        "能理解，很多家长一开始都会先担心这个。",
    ],
    "complaint": [
        "理解您现在着急，这种情况确实会影响体验。",
        "抱歉让您遇到这种情况了，我先帮您尽快登记。",
        "这个我理解，我们先按处理流程走，我先把信息收齐。",
    ],
}


DISCOVERY_QUESTIONS = {
    "grade": [
        "孩子现在几年级了？",
        "我先确认一个关键信息，孩子目前是几年级？",
    ],
    "subject": [
        "现在最想优先解决哪一科？",
        "您这边更想先看哪一科，语文、数学还是英语？",
    ],
    "contact": [
        "方便留个电话或微信吗？我把试听和方案安排发给您。",
        "您给我留一个电话或微信，我这边好继续给您安排。",
    ],
    "schedule": [
        "您看这周还是下周方便先安排试听？",
        "您这边更方便这周还是下周，我先帮您预留一下时间。",
    ],
    "demand_summary": [
        "您这次最想先解决的，是提分、学习习惯，还是某一科明显薄弱？",
        "您现在最着急的是成绩、方法，还是孩子状态这块？",
    ],
    "order_no": [
        "方便把订单号或上课记录发我一下吗？这样处理会更快。",
        "您把订单号或具体上课信息给我，我先一起登记进去。",
    ],
}


OBJECTION_GUIDES = {
    "busy": [
        "理解，您这会儿忙，我就说最关键的。",
        "明白，我不多占您时间。",
    ],
    "expensive": [
        "理解您会先比价格，这很正常。",
        "明白，费用这块肯定要算清楚。",
    ],
    "far": [
        "距离这个顾虑很正常。",
        "这个我理解，很多家长一开始也先担心距离。",
    ],
    "no_need": [
        "明白，我不会硬推您。",
        "可以理解，如果暂时没打算我就不跟您绕了。",
    ],
    "already_have": [
        "明白，已经在补课说明您本身就很重视孩子学习。",
        "理解，已经有老师不代表就不能再帮您判断一下效果。",
    ],
    "child_unwilling": [
        "这个很常见，孩子不配合确实是家长最头疼的一类情况。",
        "能理解，孩子不愿意学时，光靠硬推通常效果也不好。",
    ],
}


TOPIC_KEYWORDS = {
    "location": ("在哪", "地址", "校区", "离", "位置"),
    "price": ("价格", "报价", "费用", "多少钱", "收费"),
    "trial": ("试听", "试课", "体验"),
    "teacher": ("老师", "师资", "教研"),
    "online": ("线上", "在线", "网课"),
    "effect": ("提分", "效果", "成绩", "有没有用"),
    "cooperation": ("合作", "加盟", "渠道"),
    "identity": ("你是谁", "你哪位", "你是干嘛的", "打电话干嘛"),
}


SUPPLEMENTAL_FAQ = {
    "price": [
        "报价不能脱离孩子年级、科目和上课形式单独说，不然我现在随口给您一个数，对您也不负责。",
        "费用这块要结合年级、科目和线上线下形式来定，我先帮您把情况对齐更靠谱。",
    ],
    "trial": [
        "试听可以安排，不过我建议先把孩子年级和科目对准，再匹配更合适的老师。",
        "试听这边没问题，我更建议先对一下孩子情况，再安排更有针对性的试听。",
    ],
    "teacher": [
        "老师不是随便排一个就上，重点还是跟孩子当前年级和问题匹配。",
        "师资这块可以说，但真正关键的是老师和孩子问题有没有对上。",
    ],
    "effect": [
        "提分这件事不能乱承诺，但可以先把问题定位清楚，再看怎么安排试听和方案。",
        "效果不能拍胸口保证，不过孩子问题定位得越清楚，后面的方案就越靠谱。",
    ],
    "online": [
        "如果您不方便到校，线上一对一也可以先了解，关键还是老师匹配和孩子配合度。",
        "线上是可以做的，不少家长一开始担心效果，实际更关键的是匹配度。",
    ],
    "cooperation": [
        "合作可以聊，不过得先确认您是渠道合作、资源合作还是课程采购方向。",
        "合作这块我可以先帮您登记，但得先明确您具体想聊哪种方式。",
    ],
}


REPHRASE_PREFIXES = [
    "我换个更直白的说法。",
    "我用家长更容易判断的方式再说一遍。",
    "您如果担心的是实际落地效果，我就说得更具体一点。",
]


SMALL_TALK_REDIRECTS = [
    "您直接说下最想了解哪块，我别跟您绕弯子。",
    "您告诉我您关心课程、校区、试听还是价格，我直接对着说。",
    "我在，您说重点就行，我这边给您快速判断。",
]


SYSTEM_LAYERS = [
    {
        "layer": "会话层",
        "goal": "接住用户消息，读取并保存当前会话状态。",
        "file": "telemarketing/storage.py",
        "code": "get_or_create_conversation / append_message / save_state",
    },
    {
        "layer": "抽取层",
        "goal": "从用户输入里抽取年级、科目、联系方式、订单号等结构化字段。",
        "file": "telemarketing/knowledge_base.py",
        "code": "extract_grade / extract_subjects / extract_phone / extract_wechat / extract_order_no",
    },
    {
        "layer": "路由层",
        "goal": "判断当前属于身份确认、普通问答、异议处理、销售推进、投诉售后还是风险拒答。",
        "file": "telemarketing/engine.py",
        "code": "_analyze_turn / _build_response",
    },
    {
        "layer": "知识层",
        "goal": "从知识库和异议库中检索最匹配内容，优先用知识库结论支撑回复。",
        "file": "telemarketing/rag.py + telemarketing/knowledge_base.py",
        "code": "search_and_rerank / search_qa / search_objection",
    },
    {
        "layer": "LLM层",
        "goal": "调用大模型做意图判断和回复润色，在鉴权失败时自动回退规则链路。",
        "file": "telemarketing/llm_service.py + telemarketing/engine.py",
        "code": "analyze_turn / rewrite_reply",
    },
    {
        "layer": "话术层",
        "goal": "把命中的知识、策略和销售目标组织成更像真人顾问的回复。",
        "file": "telemarketing/playbook.py + telemarketing/engine.py",
        "code": "HUMAN_PREFIXES / OBJECTION_GUIDES / _compose_*",
    },
    {
        "layer": "业务层",
        "goal": "在合适节点创建线索、工单，或者做软收口和人工转接。",
        "file": "telemarketing/storage.py + telemarketing/engine.py",
        "code": "create_lead / create_ticket / _handle_sales / _handle_complaint",
    },
]

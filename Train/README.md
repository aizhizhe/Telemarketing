# Train Data

本目录用于保存电话销售训练与回归数据，分为两类：

- `web_sources.jsonl`：公开网页来源的元数据与正文提取结果
- `web_dialogue_seeds.jsonl`：基于公开网页内容整理出的对话种子，保留来源和证据文本
- `generated_dialogue_seeds.jsonl`：基于策划文档、网页样本和业务规则扩展出的覆盖场景
- `benchmark_scenarios_500.jsonl`：最终用于回归的 500 组完整多轮场景
- `reports/`：训练和 benchmark 产物
- `raw_web/`：抓取的原始 HTML

生成方式：

```powershell
python E:\Project\Telemarketing2\Telemarketing\Train\build_train_assets.py
python E:\Project\Telemarketing2\Telemarketing\telemarketing_app\qa\run_benchmark.py
```

说明：

- 网页来源会尽量保留原始标题、URL 和提取证据。
- 部分公开文章是话术整理、行业复盘或家长投诉案例，不一定是完整逐字稿；这类样本会在数据里标记为 `reconstructed_from_public_source=true`。
- 为了补齐覆盖面，还会生成一部分扩展场景，并明确标记为 `source_type=generated`。

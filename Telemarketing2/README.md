# Telemarketing2

单 Agent 电话招生调试台，核心链路为：

`意图识别 -> 流程规划 -> 引导话术(LLM) -> 专业回复(Data) -> 拉回流程 -> 礼貌收尾`

## 目录说明

- `Data/`：专业回复素材库
- `Data2/`：流程节点与流程规则素材库
- `config/rules.json`：可编辑链路规则
- `app/`：后端代码
- `static/index.html`：多面板调试台
- `scripts/enrich_workbooks.py`：补库脚本

## 运行

```powershell
cd E:\Project\Telemarketing2\Telemarketing2
python main.py
```

默认地址：

- [http://127.0.0.1:8090](http://127.0.0.1:8090)

## 配置

默认会优先读取当前项目下的 `.env`。如果不存在，会回退读取：

- `E:/Project/Telemarketing/phonecalljg/.env`

这样可以直接复用旧工程里已经验证过的 DashScope 兼容 OpenAI 调用方式。

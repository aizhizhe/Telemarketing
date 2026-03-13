# Telemarketing MVP

这是按策划文档落地的一期单智能体电话销售 MVP。

能力范围：

- 独立知识库目录：`knowledge_base/raw`
- RAG 检索：复用 `phonecalljg` 的问答/异议评分思路，并增加可选 embedding 重排
- 会话编排：`answer / clarify / collect_lead / handoff / refuse`
- 业务闭环：线索、工单、消息、会话、知识文档写入 SQLite
- 接口：FastAPI `POST /chat`
- 本地演示：命令行 `demo_chat.py`

## 运行

```powershell
cd E:\Project\Telemarketing2\Telemarketing\telemarketing_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

命令行演示：

```powershell
python demo_chat.py
```

单元测试：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

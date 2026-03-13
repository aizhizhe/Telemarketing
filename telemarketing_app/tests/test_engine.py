from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telemarketing.engine import TelemarketingEngine
from telemarketing.knowledge_base import KnowledgeBase
from telemarketing.settings import Settings
from telemarketing.storage import TelemarketingStorage


class TelemarketingEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "telemarketing.db"
        settings = Settings(
            knowledge_base_dir=PROJECT_ROOT / "knowledge_base" / "raw",
            database_path=db_path,
            api_key="",
        )
        knowledge_base = KnowledgeBase(settings.knowledge_base_dir)
        storage = TelemarketingStorage(settings.database_path)
        self.engine = TelemarketingEngine(
            settings=settings,
            knowledge_base=knowledge_base,
            storage=storage,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_faq_uses_knowledge_base(self) -> None:
        result = self.engine.chat(
            user_text="你们在哪啊？",
            external_user_id="u1",
            session_key="s1",
            channel="phone",
        )
        self.assertEqual(result["reply_type"], "answer")
        self.assertIn("海淀", result["reply"])

    def test_sales_lead_collects_contact(self) -> None:
        first = self.engine.chat(
            user_text="我想了解一下报价",
            external_user_id="u2",
            session_key="s2",
            channel="phone",
        )
        self.assertEqual(first["intent"], "sales_lead")
        second = self.engine.chat(
            user_text="孩子初二数学，微信是abc12345，想约试听",
            external_user_id="u2",
            session_key="s2",
            channel="phone",
        )
        self.assertEqual(second["reply_type"], "collect_lead")
        self.assertIn("跟进", second["reply"])

    def test_ticket_flow_creates_handoff(self) -> None:
        first = self.engine.chat(
            user_text="我要投诉，老师老是迟到",
            external_user_id="u3",
            session_key="s3",
            channel="phone",
        )
        self.assertEqual(first["intent"], "complaint_after_sales")
        second = self.engine.chat(
            user_text="电话是13800138000，订单号A12345678",
            external_user_id="u3",
            session_key="s3",
            channel="phone",
        )
        self.assertEqual(second["reply_type"], "handoff")
        self.assertIn("工单号", second["reply"])


if __name__ == "__main__":
    unittest.main()

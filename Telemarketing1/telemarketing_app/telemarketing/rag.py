from __future__ import annotations

import math
from functools import lru_cache

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .knowledge_base import KnowledgeBase
from .models import RetrievalHit
from .settings import Settings, get_settings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRAGService:
    def __init__(self, knowledge_base: KnowledgeBase, settings: Settings | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.settings = settings or get_settings()
        self._client = None
        if OpenAI is not None and self.settings.api_key:
            self._client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url)

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None

    def search_and_rerank(
        self,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        preferred_type: str | None = None,
    ) -> list[RetrievalHit]:
        coarse_hits = self.knowledge_base.search(
            query,
            top_k=top_k or self.settings.top_k,
            preferred_type=preferred_type,
        )
        if not coarse_hits:
            return []
        top_n = top_n or self.settings.top_n
        if not self._client or len(coarse_hits) == 1:
            return coarse_hits[:top_n]
        try:
            query_vector = self._embed(query)
            doc_vectors = self._embed_many([hit.text for hit in coarse_hits])
        except Exception:
            return coarse_hits[:top_n]
        reranked: list[RetrievalHit] = []
        for hit, vector in zip(coarse_hits, doc_vectors):
            dense_score = max(_cosine_similarity(query_vector, vector), 0.0)
            final_score = hit.score + dense_score * 5.0
            reranked.append(
                RetrievalHit(
                    kb_type=hit.kb_type,
                    title=hit.title,
                    question=hit.question,
                    answer=hit.answer,
                    scene=hit.scene,
                    keywords=hit.keywords,
                    source_file=hit.source_file,
                    text=hit.text,
                    score=final_score,
                    matched_variant=hit.matched_variant,
                    rerank_score=dense_score,
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_n]

    @lru_cache(maxsize=128)
    def _embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=[text],
            dimensions=self.settings.embedding_dim,
        )
        return response.data[0].embedding

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dim,
        )
        return [item.embedding for item in response.data]

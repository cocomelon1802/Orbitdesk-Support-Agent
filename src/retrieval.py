"""
Retrieval over the OrbitDesk knowledge base + resolved cases.

No managed vector DB is used, per the assignment ("A managed vector
database is not required"). The corpus is tiny (10 KB docs + 8 cases) so
an in-memory numpy/sklearn similarity search is sufficient and keeps the
whole thing dependency-light and fast to load.

Chunking strategy:
- KB markdown docs are split on "## " headings so each chunk is one
  coherent sub-topic, tagged with the document_id from the YAML front matter.
- Each resolved case becomes a single chunk (they're already short and
  self-contained), tagged with its case_id and status (resolved / escalated
  / superseded). Superseded cases are kept in the index -- the assignment
  explicitly wants them retrievable for testing -- but are flagged so
  downstream nodes never present them as current guidance.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .models import EmbeddingModel


@dataclass
class Chunk:
    source_id: str
    doc_status: str  # current | resolved | escalated | superseded
    title: str
    text: str


def _parse_front_matter(md_text: str) -> Dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", md_text, re.DOTALL)
    meta = {}
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def _chunk_kb_doc(path: Path) -> List[Chunk]:
    text = path.read_text(encoding="utf-8")
    meta = _parse_front_matter(text)
    doc_id = meta.get("document_id", path.stem)
    status = meta.get("status", "current")
    title = meta.get("title", path.stem)

    body = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    sections = re.split(r"\n(?=## )", body)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        chunks.append(Chunk(source_id=doc_id, doc_status=status, title=title, text=section))
    return chunks


def _chunk_resolved_cases(path: Path) -> List[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for case in data.get("cases", []):
        parts = [f"Title: {case['title']}", f"Status: {case['status']}"]
        if case.get("symptoms"):
            parts.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            parts.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            parts.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            parts.append("Superseded reason: " + case["superseded_reason"])
        text = "\n".join(parts)
        chunks.append(
            Chunk(
                source_id=case["case_id"],
                doc_status=case["status"],
                title=case["title"],
                text=text,
            )
        )
    return chunks


class RetrievalIndex:
    def __init__(self, data_dir: str, embedder: EmbeddingModel):
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self._matrix = None
        self._build()

    def _build(self):
        for md_path in sorted(self.data_dir.glob("*.md")):
            self.chunks.extend(_chunk_kb_doc(md_path))

        cases_path = self.data_dir / "resolved_cases.json"
        if cases_path.exists():
            self.chunks.extend(_chunk_resolved_cases(cases_path))

        texts = [c.text for c in self.chunks]
        self._matrix = self.embedder.encode_corpus(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embedder.encode([query])

        if self.embedder.backend == "sentence-transformers":
            sims = np.dot(self._matrix, np.asarray(q_vec)[0])
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            sims = cosine_similarity(q_vec, self._matrix)[0]

        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            c = self.chunks[idx]
            results.append(
                {
                    "source_id": c.source_id,
                    "doc_status": c.doc_status,
                    "title": c.title,
                    "passage": c.text,
                    "score": float(sims[idx]),
                }
            )
        return results

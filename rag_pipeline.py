from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_SOURCE_FILES = [
    "decision_engine.py",
    "ml_pipeline.py",
    "README.md",
    "medicine_facts.md",
]
# Additional operational knowledge files (mock recall/SOP docs) are read from this folder.
SAFETY_DOCS_DIR = "safety_docs"
SAFETY_DOC_GLOBS = ("*.md", "*.txt")


@dataclass
class RetrievedChunk:
    # Minimal unit returned by retrieval: where it came from, matched text, and relevance score.
    source: str
    text: str
    score: float


def _read_text(path: Path) -> str:
    # Soft-fail if file does not exist to keep corpus build resilient.
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    # Split long docs into overlapping windows so retrieval can match local context.
    if not text.strip():
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _iter_source_paths(project_root: Path) -> List[Path]:
    # Start with core project files, then append any safety docs discovered on disk.
    paths: List[Path] = []
    for rel_path in DEFAULT_SOURCE_FILES:
        paths.append(project_root / rel_path)

    safety_root = project_root / SAFETY_DOCS_DIR
    if safety_root.exists():
        for pattern in SAFETY_DOC_GLOBS:
            for path in sorted(safety_root.glob(pattern)):
                paths.append(path)

    # Keep deterministic order while avoiding duplicates.
    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    return deduped


def build_local_corpus(project_root: Path) -> List[Dict[str, str]]:
    # Build chunked corpus used by local retriever.
    docs: List[Dict[str, str]] = []
    for file_path in _iter_source_paths(project_root):
        rel_path = str(file_path.relative_to(project_root)) if file_path.exists() else file_path.name
        text = _read_text(file_path)
        for idx, chunk in enumerate(_chunk_text(text)):
            docs.append(
                {
                    "source": rel_path,
                    "chunk_id": f"{rel_path}::chunk_{idx}",
                    "text": chunk,
                }
            )
    return docs


class BasicLocalRetriever:
    def __init__(self, corpus_docs: List[Dict[str, str]]):
        if not corpus_docs:
            raise ValueError("Corpus is empty; cannot initialize retriever.")
        self.corpus_docs = corpus_docs
        # TF-IDF keeps retrieval fully local and lightweight (no external API dependency).
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([d["text"] for d in corpus_docs])

    def search(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        # Rank corpus chunks by cosine similarity and return top-k matches.
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results: List[RetrievedChunk] = []
        for idx in top_idx:
            doc = self.corpus_docs[int(idx)]
            results.append(
                RetrievedChunk(
                    source=doc["source"],
                    text=doc["text"],
                    score=float(sims[int(idx)]),
                )
            )
        return results


def detect_intent(question: str) -> str:
    # Simple intent router used by the agent loop before retrieval/answering.
    q = question.lower()
    if any(
        word in q
        for word in [
            "recall",
            "batch",
            "fda",
            "supplier notice",
            "safety",
            "quarantine",
            "damaged packaging",
            "expiry",
            "storage warning",
            "temperature",
        ]
    ):
        return "recall_safety"
    if any(word in q for word in ["low", "critical", "risk", "urgent"]):
        return "low_stock"
    if any(word in q for word in ["forecast", "predict", "demand", "tomorrow", "next"]):
        return "forecast"
    if any(word in q for word in ["transfer", "reorder", "decision", "cost", "savings"]):
        return "decision"
    return "general"


def _extract_batch_id(text: str) -> str:
    # Detect batch-like tokens (e.g., PCM-042) from user question.
    match = re.search(r"\b[A-Z]{2,5}-\d{2,4}\b", text.upper())
    return match.group(0) if match else ""


def collect_store_runtime_facts(
    store_name: str,
    owner_inventory: pd.DataFrame,
    owner_low: pd.DataFrame,
    owner_forecast: pd.DataFrame,
    owner_recommendations: pd.DataFrame,
    selected_medicine: str | None = None,
    medicine_fact: str | None = None,
) -> Dict[str, str]:
    # Gather store-scoped runtime facts so responses are grounded in current analysis output.
    top_low = owner_low.sort_values("daily_demand", ascending=False).head(3)
    top_low_text = ", ".join(top_low["medicine_name"].astype(str).tolist()) or "None"

    forecast_stats = "No forecast rows"
    if not owner_forecast.empty:
        forecast_stats = (
            f"rows={len(owner_forecast)}, avg_predicted_demand="
            f"{owner_forecast['predicted_demand'].mean():.2f}"
        )

    transfer_count = 0
    reorder_count = 0
    if not owner_recommendations.empty:
        transfer_count = int((owner_recommendations["final_decision"] == "Transfer Stock").sum())
        reorder_count = int((owner_recommendations["final_decision"] == "Reorder Stock").sum())

    return {
        "store_name": store_name,
        "inventory_rows": str(len(owner_inventory)),
        "low_stock_rows": str(len(owner_low)),
        "top_low_medicines": top_low_text,
        "forecast_summary": forecast_stats,
        "transfer_recommendations": str(transfer_count),
        "reorder_recommendations": str(reorder_count),
        "selected_medicine": selected_medicine or "All Medicines",
        "medicine_fact": medicine_fact or "",
    }


def answer_with_evidence(
    question: str,
    intent: str,
    runtime_facts: Dict[str, str],
    retrieved: List[RetrievedChunk],
) -> Dict[str, object]:
    # Build concise response text + actionable next step + evidence snippets.
    store_name = runtime_facts["store_name"]
    selected_medicine = runtime_facts.get("selected_medicine", "All Medicines")
    medicine_fact = runtime_facts.get("medicine_fact", "")
    answer = (
        f"For {store_name}, there are {runtime_facts['low_stock_rows']} low/critical items. "
        f"Top low medicines: {runtime_facts['top_low_medicines']}."
    )
    next_step = "Review the low-stock table and prioritize medicines with highest daily demand."

    if intent == "forecast":
        answer = (
            f"Forecast summary for {store_name}: {runtime_facts['forecast_summary']}. "
            "Use this to confirm reorder urgency for low-stock medicines."
        )
        next_step = "Match forecast demand against current stock and confirm reorder quantities."
    elif intent == "decision":
        answer = (
            f"Decision queue for {store_name}: "
            f"{runtime_facts['transfer_recommendations']} transfer suggestions and "
            f"{runtime_facts['reorder_recommendations']} reorder suggestions."
        )
        next_step = "Execute transfer recommendations first when savings and availability are favorable."
    elif intent == "general":
        answer = (
            f"Current state for {store_name}: inventory rows={runtime_facts['inventory_rows']}, "
            f"low-stock rows={runtime_facts['low_stock_rows']}, {runtime_facts['forecast_summary']}."
        )
        next_step = "Ask about low stock, forecast, or transfer/reorder decisions for targeted guidance."
    elif intent == "recall_safety":
        # Safety/recall mode: infer decision from retrieved policy/notice keywords.
        combined = " ".join(chunk.text.lower() for chunk in retrieved)
        batch_id = _extract_batch_id(question)
        decision = "Hold & Verify"
        reason = "No explicit recall match found in current retrieved safety notices."
        action = (
            "Temporarily hold sale, re-check latest supplier/FDA notice, and escalate to manager if uncertain."
        )
        if "do not sell" in combined or "recalled" in combined or "quarantine" in combined:
            decision = "Do Not Sell"
            reason = "Retrieved safety documents indicate recall/quarantine conditions."
            action = (
                "Remove from shelf, quarantine inventory, notify manager, and offer alternate non-affected batch."
            )
        elif "safe to sell" in combined:
            decision = "Safe to Sell (Policy Conditions)"
            reason = "Retrieved policy text indicates sale is allowed under stated conditions."
            action = "Confirm packaging integrity and expiry threshold before sale."

        batch_txt = f" Batch checked: {batch_id}." if batch_id else ""
        answer = f"{decision}.{batch_txt} Reason: {reason}"
        next_step = action

    if selected_medicine and selected_medicine != "All Medicines":
        # If user filtered medicine, keep response explicitly medicine-scoped.
        answer = f"{answer} Selected medicine scope: {selected_medicine}."
        if medicine_fact:
            answer = f"{answer} Key handling fact: {medicine_fact}"
            next_step = (
                f"{next_step} Also verify storage/handling controls for {selected_medicine} "
                "before transfer or reorder execution."
            )

    evidence = []
    for chunk in retrieved:
        # Return short evidence excerpts for transparency and demo explainability.
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180] + "..."
        evidence.append(f"{chunk.source}: {snippet}")

    return {
        "answer": answer,
        "evidence": evidence,
        "next_step": next_step,
    }

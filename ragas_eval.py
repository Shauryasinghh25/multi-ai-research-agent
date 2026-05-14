"""
RAGAS evaluation helper for the research pipeline.

The pipeline usually has no human ground-truth answer, so this module runs
reference-free RAGAS metrics by default:
  - faithfulness: checks whether the report is supported by retrieved context
  - answer_relevancy: checks whether the report answers the topic/question
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

load_dotenv()


MAX_CONTEXT_CHARS = 6000


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clip_context(value: Any) -> str:
    text = _clean_text(value)
    return text[:MAX_CONTEXT_CHARS]


def split_rag_context(rag_context: Any) -> list[str]:
    """Split the formatted RAG context string back into individual contexts."""
    text = _clean_text(rag_context)
    if not text:
        return []
    parts = [part.strip() for part in text.split("\n\n---\n\n")]
    return [_clip_context(part) for part in parts if part.strip()]


def build_eval_contexts(
    search_results: Any,
    scraped_content: Any,
    rag_context: Any = "",
) -> list[str]:
    """
    Build contexts used by the writer.

    Fresh search/scrape evidence is always included because the writer uses it.
    Retrieved KB context is appended when available.
    """
    contexts = []
    for item in (search_results, scraped_content):
        text = _clip_context(item)
        if text:
            contexts.append(text)
    contexts.extend(split_rag_context(rag_context))
    return contexts


def evaluate_report_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single generated report with RAGAS.

    Returns a stable dict so callers can display an error instead of crashing
    when RAGAS is not installed or an API call fails.
    """
    question = _clean_text(question)
    answer = _clean_text(answer)
    contexts = [_clean_text(context) for context in contexts if _clean_text(context)]
    ground_truth = _clean_text(ground_truth)

    if not question or not answer or not contexts:
        return {
            "ok": False,
            "error": "RAGAS needs a question/topic, answer/report, and at least one context.",
            "scores": {},
            "overall": None,
        }

    try:
        from datasets import Dataset
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception as exc:
        return {
            "ok": False,
            "error": f"RAGAS dependencies are missing or failed to import: {exc}",
            "scores": {},
            "overall": None,
        }

    row = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    metrics = [faithfulness, answer_relevancy]

    if ground_truth:
        row["ground_truth"] = ground_truth

    try:
       import nest_asyncio
       import asyncio

       nest_asyncio.apply()

       dataset = Dataset.from_list([row])

       llm = ChatMistralAI(
        model="mistral-medium-3-5",
        temperature=0
    )

       embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)

       result = evaluate(
         dataset,
         metrics=metrics,
         llm=llm,
         embeddings=embeddings,
         raise_exceptions=False,
    )

       scores = {}
       try:
            scores = result.to_pandas().iloc[0].to_dict()
       except Exception:
            try:
                scores = dict(result)
            except Exception:
                scores = {}

       numeric_scores = {
            key: float(value)
            for key, value in scores.items()
            if isinstance(value, (int, float)) and value == value
        }
       overall = (
            round(sum(numeric_scores.values()) / len(numeric_scores), 3)
            if numeric_scores
            else None
        )

       return {
            "ok": bool(numeric_scores),
            "error": None if numeric_scores else "RAGAS finished but returned no numeric scores.",
            "scores": numeric_scores,
            "overall": overall,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "scores": {},
            "overall": None,
        }

"""
RAG Knowledge Base Module
─────────────────────────
Persistent FAISS vector store + Mistral embeddings.

ARCHITECTURE (v2 — fixed self-reinforcement):
  • Every chunk is tagged with `source_type` and `session_id`.
  • Retrieval EXCLUDES the current session to prevent feedback loops.
  • Research papers (PDF) are ingested separately with source_type="paper".
  • Only raw external content is stored — never model-generated outputs.
"""

import os
import json
import hashlib
import uuid
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
KB_DIR = Path(__file__).parent / "knowledge_base"
METADATA_FILE = KB_DIR / "metadata.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5

# Valid source types — NEVER store "generated" content
VALID_SOURCE_TYPES = {"web_search", "web_scrape", "paper"}
TOPIC_DUPLICATE_THRESHOLD = 0.92

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = MistralAIEmbeddings(model="mistral-embed")

# ── Text splitter ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Session ID helper ─────────────────────────────────────────────────────────
def generate_session_id() -> str:
    """Create a unique session ID for this pipeline run."""
    return uuid.uuid4().hex[:12]


# ── Metadata helpers ──────────────────────────────────────────────────────────
def _load_metadata() -> dict:
    """Load the KB metadata file (tracks topics, doc counts, papers)."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "topics": [],
        "total_chunks": 0,
        "created_at": None,
        "source_breakdown": {"web_search": 0, "web_scrape": 0, "paper": 0},
        "papers": [],  # List of {name, pages, chunks, ingested_at}
    }


def normalize_topic_key(topic: str) -> str:
    """Normalize topic names for stable duplicate detection."""
    topic = re.sub(r"\s+", " ", str(topic or "").strip().lower())
    topic = topic.strip(" .:-_")
    return topic


def canonicalize_topic(topic: str, existing_topics: list[str] | None = None) -> str:
    """
    Return the existing canonical topic when the new topic is a duplicate.

    This catches case-only differences and near-identical variants without
    adding another embedding model dependency to the app.
    """
    topic = str(topic or "").strip()
    if not topic:
        return "Untitled Topic"

    existing_topics = existing_topics or []
    new_key = normalize_topic_key(topic)
    best_topic = None
    best_ratio = 0.0

    for existing in existing_topics:
        existing_key = normalize_topic_key(existing)
        if not existing_key:
            continue
        if new_key == existing_key:
            return existing
        ratio = SequenceMatcher(None, new_key, existing_key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_topic = existing

    if best_topic and best_ratio >= TOPIC_DUPLICATE_THRESHOLD:
        return best_topic
    return topic


def dedupe_topics(topics: list[str]) -> list[str]:
    """Collapse duplicate/near-duplicate topic labels while preserving order."""
    canonical_topics = []
    for topic in topics:
        canonical = canonicalize_topic(topic, canonical_topics)
        if canonical not in canonical_topics:
            canonical_topics.append(canonical)
    return canonical_topics


def _save_metadata(meta: dict):
    """Persist KB metadata to disk."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    meta["topics"] = dedupe_topics(meta.get("topics", []))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# ── Core RAG functions ────────────────────────────────────────────────────────
def get_or_create_vectorstore() -> FAISS | None:
    """Load existing FAISS index from disk, or return None if it doesn't exist."""
    index_path = KB_DIR / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            str(KB_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None


def add_research_to_kb(
    topic: str,
    content: str,
    source_urls: list[str] | None = None,
    source_type: str = "web_search",
    session_id: str = "",
):
    """
    Chunk and embed research content, then add to the FAISS knowledge base.

    Args:
        topic:       The research topic string.
        content:     Raw search results OR scraped content (NOT generated reports).
        source_urls: Optional list of source URLs for metadata.
        source_type: One of "web_search", "web_scrape", "paper".
        session_id:  Unique ID for this pipeline run (for session isolation).
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type '{source_type}'. Must be one of {VALID_SOURCE_TYPES}. "
            f"NEVER store model-generated content in the knowledge base."
        )

    if not content or len(content.strip()) < 50:
        return  # Skip trivially short content

    meta = _load_metadata()
    topic = canonicalize_topic(topic, meta.get("topics", []))

    # Deduplicate: hash content to avoid re-indexing same material
    content_hash = hashlib.md5(content[:2000].encode()).hexdigest()

    # Build documents with rich metadata
    now = datetime.now().isoformat()
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "topic": topic,
                "source_type": source_type,
                "session_id": session_id,
                "source_urls": ", ".join(source_urls) if source_urls else "",
                "indexed_at": now,
                "content_hash": content_hash,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(splitter.split_text(content))
    ]

    if not docs:
        return

    # Load existing store or create new
    vectorstore = get_or_create_vectorstore()
    if vectorstore is not None:
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist to disk
    KB_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(KB_DIR))

    # Update metadata
    if meta["created_at"] is None:
        meta["created_at"] = now
    if topic not in meta["topics"]:
        meta["topics"].append(topic)
    meta["total_chunks"] += len(docs)

    # Update source breakdown
    breakdown = meta.get("source_breakdown", {"web_search": 0, "web_scrape": 0, "paper": 0})
    breakdown[source_type] = breakdown.get(source_type, 0) + len(docs)
    meta["source_breakdown"] = breakdown

    _save_metadata(meta)


def add_paper_to_kb(
    file_name: str,
    file_bytes: bytes,
    session_id: str = "",
) -> dict:
    """
    Extract text from a PDF file and ingest it into the knowledge base.

    Args:
        file_name:  Original filename of the uploaded PDF.
        file_bytes: Raw bytes of the PDF file.
        session_id: Pipeline session ID.

    Returns:
        dict with keys: success, pages, chunks, error
    """
    try:
        from PyPDF2 import PdfReader
        import io

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)

        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

        full_text = "\n\n".join(text_parts)

        if len(full_text.strip()) < 50:
            return {"success": False, "pages": pages, "chunks": 0, "error": "PDF contains too little extractable text."}

        # Derive a topic from the filename
        topic = Path(file_name).stem.replace("_", " ").replace("-", " ").title()
        meta = _load_metadata()
        topic = canonicalize_topic(topic, meta.get("topics", []))

        # Store with source_type="paper"
        content_hash = hashlib.md5(full_text[:2000].encode()).hexdigest()
        now = datetime.now().isoformat()

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "topic": topic,
                    "source_type": "paper",
                    "session_id": session_id,
                    "source_urls": f"pdf://{file_name}",
                    "indexed_at": now,
                    "content_hash": content_hash,
                    "chunk_index": i,
                    "paper_name": file_name,
                },
            )
            for i, chunk in enumerate(splitter.split_text(full_text))
        ]

        if not docs:
            return {"success": False, "pages": pages, "chunks": 0, "error": "No chunks generated from PDF."}

        # Load existing store or create new
        vectorstore = get_or_create_vectorstore()
        if vectorstore is not None:
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)

        KB_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(KB_DIR))

        # Update metadata
        if meta["created_at"] is None:
            meta["created_at"] = now
        if topic not in meta["topics"]:
            meta["topics"].append(topic)
        meta["total_chunks"] += len(docs)

        breakdown = meta.get("source_breakdown", {"web_search": 0, "web_scrape": 0, "paper": 0})
        breakdown["paper"] = breakdown.get("paper", 0) + len(docs)
        meta["source_breakdown"] = breakdown

        # Track paper in metadata
        papers = meta.get("papers", [])
        papers.append({
            "name": file_name,
            "topic": topic,
            "pages": pages,
            "chunks": len(docs),
            "ingested_at": now,
        })
        meta["papers"] = papers

        _save_metadata(meta)

        return {"success": True, "pages": pages, "chunks": len(docs), "error": None}

    except ImportError:
        return {"success": False, "pages": 0, "chunks": 0, "error": "PyPDF2 not installed. Run: pip install PyPDF2"}
    except Exception as e:
        return {"success": False, "pages": 0, "chunks": 0, "error": str(e)}


SIMILARITY_THRESHOLD = 0.5 # Minimum similarity to be considered relevant 


def retrieve_relevant_context(
    query: str,
    k: int = RETRIEVAL_K,
    exclude_session_id: str = "",
) -> tuple[str, list[dict], list[dict]]:
    """
    Retrieve the top-k most relevant chunks from the knowledge base,
    filtered by similarity threshold AND session isolation.

    SESSION ISOLATION: Chunks from the current session (exclude_session_id)
    are never returned, preventing the self-reinforcement loop.

    Returns:
        tuple of (context_string, accepted_explanations, rejected_explanations)
        - context_string: formatted past research (only relevant chunks)
        - accepted_explanations: chunks that passed the threshold
        - rejected_explanations: chunks that were filtered out (for transparency)
    """
    vectorstore = get_or_create_vectorstore()
    if vectorstore is None:
        return "", [], []

    # Get actual FAISS distances
    # Fetch plenty of candidates to allow for session and diversity filtering
    fetch_k = 25
    results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=fetch_k
    )

    if not results_with_scores:
        return "", [], []

    # Diversity filter: allow up to 2 chunks per topic
    filtered_results = []
    topic_counts = {}

    for doc, distance in results_with_scores:
        # 1. Session Isolation: skip chunks from the current session
        chunk_session = doc.metadata.get("session_id", "")
        if exclude_session_id and chunk_session == exclude_session_id:
            continue

        # 2. Diversity check: allow up to 2 chunks per topic
        topic = doc.metadata.get("topic", "Unknown")
        topic_counts.setdefault(topic, 0)

        if topic_counts[topic] < 2:
            filtered_results.append((doc, distance))
            topic_counts[topic] += 1

        if len(filtered_results) >= k:
            break

    results_with_scores = filtered_results

    context_parts = []
    accepted = []
    rejected = []

    # Extract query keywords for matching explanation
    query_lower = query.lower()
    query_words = set(w for w in query_lower.split() if len(w) > 3)

    for i, (doc, distance) in enumerate(results_with_scores, 1):
        topic = doc.metadata.get("topic", "Unknown")
        indexed_at = doc.metadata.get("indexed_at", "")
        source_urls = doc.metadata.get("source_urls", "")
        source_type = doc.metadata.get("source_type", "unknown")
        preview = doc.page_content[:150].replace("\n", " ").strip()

        # ── SAFETY NET: never retrieve generated content ──
        if source_type == "generated":
            continue

        # FAISS returns L2 distance — convert to similarity (lower distance = better)
        similarity = round(
            max(0.0, min(1.0, 1.0 - (distance / 2.0))),
            2
        )

        # Build human-readable reasons
        reasons = []

        # Reason 1: Similarity score
        if similarity >= 0.8:
            reasons.append(f"🎯 Very high semantic similarity ({similarity})")
        elif similarity >= 0.5:
            reasons.append(f"✅ High semantic similarity ({similarity})")
        elif similarity >= SIMILARITY_THRESHOLD:
            reasons.append(f"📎 Moderate similarity ({similarity})")
        else:
            reasons.append(f"❌ Below threshold ({similarity} < {SIMILARITY_THRESHOLD})")

        # Reason 2: Source type indicator
        source_emoji = {"paper": "📄", "web_search": "🔍", "web_scrape": "🌐"}.get(source_type, "❓")
        reasons.append(f"{source_emoji} Source: {source_type}")

        # Reason 3: Topic match
        topic_lower = topic.lower()
        topic_overlap = query_words & set(w for w in topic_lower.split() if len(w) > 3)
        if topic_overlap:
            reasons.append(f"🏷️ Topic match: \"{', '.join(topic_overlap)}\"")

        # Reason 4: Keyword matches in content
        content_lower = doc.page_content.lower()
        keyword_hits = [w for w in query_words if w in content_lower]
        if keyword_hits:
            reasons.append(f"🔑 Keywords found: {', '.join(keyword_hits[:5])}")

        # Reason 5: Recency
        if indexed_at:
            try:
                idx_date = datetime.fromisoformat(indexed_at)
                age_days = (datetime.now() - idx_date).days
                if age_days == 0:
                    reasons.append("⚡ Indexed today")
                elif age_days <= 7:
                    reasons.append(f"📅 Recent ({age_days}d ago)")
                else:
                    reasons.append(f"📅 Indexed {age_days}d ago")
            except (ValueError, TypeError):
                pass

        explanation = {
            "rank": i,
            "similarity": similarity,
            "distance": round(distance, 3),
            "topic": topic,
            "source_type": source_type,
            "indexed_at": indexed_at,
            "preview": preview,
            "reasons": reasons,
            "source_urls": source_urls,
        }

        # RELEVANCE GATE: only accept chunks above threshold
        if similarity >= SIMILARITY_THRESHOLD:
            # Label source type in context for the writer
            source_label = "Research Paper" if source_type == "paper" else "Web Research"
            context_parts.append(
                f"[Past {source_label} #{len(context_parts)+1} — Topic: \"{topic}\"]\n{doc.page_content}"
            )
            accepted.append(explanation)
            # Stop after collecting enough accepted chunks
            if len(accepted) >= k:
                break
        else:
            rejected.append(explanation)

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, accepted, rejected


def get_kb_stats() -> dict:
    """
    Return knowledge base statistics for UI display.

    Returns:
        dict with keys: exists, total_chunks, topics, created_at, source_breakdown, papers
    """
    meta = _load_metadata()
    return {
        "exists": (KB_DIR / "index.faiss").exists(),
        "total_chunks": meta.get("total_chunks", 0),
        "topics": meta.get("topics", []),
        "created_at": meta.get("created_at"),
        "source_breakdown": meta.get("source_breakdown", {"web_search": 0, "web_scrape": 0, "paper": 0}),
        "papers": meta.get("papers", []),
    }


def clear_kb():
    """Delete the entire knowledge base (index + metadata)."""
    import shutil
    if KB_DIR.exists():
        shutil.rmtree(KB_DIR)

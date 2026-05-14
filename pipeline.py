from agents import build_reader_agent , build_search_agent , writer_chain , critic_chain
from rag import add_research_to_kb, retrieve_relevant_context, get_kb_stats, generate_session_id
from ragas_eval import build_eval_contexts, evaluate_report_with_ragas
import re

def _extract_urls(text: str) -> list[str]:
    """Extract URLs from text content."""
    return re.findall(r'https?://[^\s\)\"\']+', text)


def _scrape_is_usable(text: str) -> bool:
    """Return False for scraper/access failure messages."""
    text = str(text or "").strip()
    bad_markers = [
        "could not scrape url",
        "could not extract",
        "provided urls either returned",
        "access errors",
        "no extractable text",
        "not extract enough clean text",
    ]
    return len(text) >= 120 and not any(marker in text.lower() for marker in bad_markers)

def run_research_pipeline(topic : str) -> dict:

    state = {}

    # Generate a unique session ID for this pipeline run
    # This prevents the self-reinforcement loop: current-session data
    # is never retrieved as "past research"
    session_id = generate_session_id()
    state["session_id"] = session_id

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    state["scrape_usable"] = _scrape_is_usable(state["scraped_content"])

    print("\nscraped content: \n", state['scraped_content'])

    #step 2.5 - RAG: detect relevant existing knowledge before storing this run
    print("\n"+" ="*50)
    print("step 2.5 - RAG Engine: detecting relevant prior knowledge ...")
    print("="*50)

    source_urls = _extract_urls(state['search_results'])

    # Retrieve relevant past research BEFORE storing current-session evidence.
    # This keeps the system honest about whether it already had knowledge.
    rag_context, rag_accepted, rag_rejected = retrieve_relevant_context(topic)
    state["rag_context"] = rag_context
    state["rag_explanations"] = rag_accepted
    state["rag_rejected"] = rag_rejected
    state["rag_used"] = len(rag_accepted) > 0
    state["knowledge_status"] = (
        "relevant_knowledge_found" if state["rag_used"] else "no_relevant_knowledge"
    )

    # Store search results and scraped content AFTER detection for future runs.
    add_research_to_kb(
        topic,
        state['search_results'],
        source_urls,
        source_type="web_search",
        session_id=session_id,
    )
    if state["scrape_usable"]:
        add_research_to_kb(
            topic,
            state['scraped_content'],
            source_urls,
            source_type="web_scrape",
            session_id=session_id,
        )

    kb_stats = get_kb_stats()
    print(f"\nKB Stats: {kb_stats['total_chunks']} chunks across {len(kb_stats['topics'])} topics")
    breakdown = kb_stats.get("source_breakdown", {})
    print(f"  Breakdown: {breakdown.get('web_search', 0)} search | {breakdown.get('web_scrape', 0)} scrape | {breakdown.get('paper', 0)} paper")
    print(f"  Accepted: {len(rag_accepted)} chunks | Rejected: {len(rag_rejected)} chunks (below threshold)")
    if rag_accepted:
        print(f"\nRelevant past context found ({len(rag_context)} chars):")
        for exp in rag_accepted:
            print(f"  ✅ #{exp['rank']} [{exp['similarity']}] {exp['topic']} ({exp.get('source_type', '?')})")
    else:
        print("\n⚠️ No relevant knowledge found → using base LLM")

    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n "
        f"{state['scraped_content'] if state['scrape_usable'] else 'Scrape unavailable; rely on search snippets and URLs only.'}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined,
        "rag_context": rag_context if state["rag_used"] else "No relevant past research available for this topic.",
    })

    print("\n Final Report\n",state['report'])

    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    #step 5 - RAGAS evaluation

    print("\n"+" ="*50)
    print("step 5 - RAGAS is evaluating the report ")
    print("="*50)

    if state["rag_used"]:
        state["ragas_eval"] = evaluate_report_with_ragas(
            question=topic,
            answer=state.get("report", ""),
            contexts=build_eval_contexts(
                state.get("search_results", ""),
                state.get("scraped_content", ""),
                state.get("rag_context", ""),
            ),
        )
    else:
        state["ragas_eval"] = {
            "ok": False,
            "skipped": True,
            "error": "No relevant RAG chunks were used, so RAGAS was skipped to avoid a misleading score.",
            "scores": {},
            "overall": None,
        }

    if state["ragas_eval"].get("ok"):
        print("\n RAGAS eval score \n", state["ragas_eval"])
    elif state["ragas_eval"].get("skipped"):
        print("\n RAGAS skipped \n", state["ragas_eval"].get("error"))
    else:
        print("\n RAGAS eval failed \n", state["ragas_eval"].get("error"))

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)

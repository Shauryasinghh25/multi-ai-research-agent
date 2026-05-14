"""
Research Agent – Streamlit UI (with RAG)
Run with:  streamlit run app.py

Requires agents.py exposing build_search_agent, build_reader_agent,
writer_chain, critic_chain.
"""

import re
import streamlit as st

try:
    from agents import (
        build_reader_agent,
        build_search_agent,
        writer_chain,
        critic_chain,
    )
    from rag import (
        add_research_to_kb,
        add_paper_to_kb,
        retrieve_relevant_context,
        get_kb_stats,
        clear_kb,
        generate_session_id,
        dedupe_topics,
    )
    from ragas_eval import build_eval_contexts, evaluate_report_with_ragas
    BACKEND_OK = True
    BACKEND_ERR = None
except Exception as e:
    BACKEND_OK = False
    BACKEND_ERR = str(e)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchGuide – Multi-Agent AI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS – Midnight Violet Design System ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg:          #04040d;
    --bg2:         #080815;
    --surface:     #0e0e1e;
    --surface2:    #141428;
    --surface3:    #1c1c34;

    --border:      rgba(139,92,246,0.08);
    --border2:     rgba(139,92,246,0.16);
    --border3:     rgba(139,92,246,0.28);

    --ink:         #ededff;
    --ink2:        #9898c0;
    --muted:       #5a5a80;

    --violet:      #8b5cf6;
    --violet2:     #7c3aed;
    --violet-dim:  rgba(139,92,246,0.10);
    --violet-glow: rgba(139,92,246,0.22);

    --emerald:     #10b981;
    --emerald-dim: rgba(16,185,129,0.10);

    --amber:       #f59e0b;
    --amber-dim:   rgba(245,158,11,0.09);
    --red:         #f87171;
    --red-dim:     rgba(248,113,113,0.09);

    --radius-lg:   18px;
    --radius-md:   12px;
    --radius-sm:   8px;
    --radius-xs:   6px;

    --shadow:      0 8px 40px rgba(0,0,0,0.55);
    --shadow-sm:   0 4px 16px rgba(0,0,0,0.4);
    --glow:        0 0 32px rgba(139,92,246,0.18);

    --grad-primary: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    --grad-hero:    linear-gradient(135deg, #8b5cf6, #a78bfa, #c4b5fd);
    --grad-bg:      radial-gradient(ellipse 70% 40% at 50% 0%, rgba(139,92,246,0.09) 0%, transparent 70%);
}

@keyframes fadeUp  { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes glow    { 0%,100%{box-shadow:0 0 0 0 rgba(139,92,246,0.4)} 50%{box-shadow:0 0 0 10px rgba(139,92,246,0)} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.45} }
@keyframes spin    { to{transform:rotate(360deg)} }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: var(--bg) !important;
    background-image: var(--grad-bg) !important;
    background-attachment: fixed !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 clamp(1rem, 4vw, 3.5rem) 5rem !important;
    max-width: 100% !important;
    margin: 0 auto !important;
}

/* ── App Header ── */
.app-header {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    animation: fadeUp 0.7s ease;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 80%; height: 220px;
    background: radial-gradient(ellipse, rgba(139,92,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.app-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--violet); margin-bottom: 1.1rem;
    display: inline-flex; align-items: center; gap: 0.6rem;
    position: relative;
}
.app-eyebrow::before, .app-eyebrow::after {
    content: ''; display: block;
    width: 30px; height: 1px;
    background: var(--violet); opacity: 0.4;
}
.app-title {
    font-size: clamp(3rem, 6vw, 5rem);
    font-weight: 900; letter-spacing: -0.04em;
    line-height: 1.0; margin-bottom: 1rem;
    position: relative;
}
.brand { color: var(--ink); }
.brand-accent {
    background: var(--grad-hero);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-subtitle {
    font-size: 1rem; color: var(--ink2); font-weight: 400;
    max-width: 580px; margin: 0 auto; line-height: 1.7;
}

/* ── Section Label ── */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin: 2rem 0 0.65rem;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px var(--violet-dim), var(--shadow-sm) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

/* ── Primary Button ── */
.stButton > button {
    background: var(--grad-primary) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    padding: 0.9rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px var(--violet-glow) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.88 !important; transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px var(--violet-glow) !important;
}
.stButton > button:disabled {
    background: var(--surface2) !important; color: var(--muted) !important;
    box-shadow: none !important; transform: none !important;
}

/* ── Pipeline Cards (now in-flow, not fixed) ── */
/* Pipeline styling removed from fixed positioning */

.pipeline-container {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    margin: 0.5rem 0 1.5rem;
    animation: fadeIn 0.5s ease;
}
.pipeline-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.pipeline-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--grad-primary); opacity: 0;
    transition: opacity 0.2s;
}
.pipeline-card:hover {
    border-color: var(--border3);
    transform: translateX(3px);
    box-shadow: var(--shadow-sm), 0 0 0 1px var(--border2);
}
.pipeline-card:hover::after { opacity: 1; }
.pipeline-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700;
    color: var(--muted); margin-bottom: 0.4rem;
    letter-spacing: 0.06em;
}
.pipeline-title {
    font-size: 0.95rem; font-weight: 700;
    color: var(--ink); margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
}
.pipeline-desc {
    font-size: 0.75rem; color: var(--ink2); line-height: 1.5;
}
.pipeline-status {
    display: inline-flex; align-items: center; gap: 0.35rem;
    margin-top: 0.7rem; padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.pipeline-status.ready {
    background: var(--emerald-dim);
    border: 1px solid rgba(16,185,129,0.2);
    color: var(--emerald);
}
.pipeline-status.running {
    background: var(--violet-dim);
    border: 1px solid rgba(139,92,246,0.2);
    color: var(--violet);
    animation: pulse 1.8s ease-in-out infinite;
}
.pipeline-status.waiting {
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--muted);
}

/* ── Toggle Button (removed, no longer needed) ── */

/* ── Expander ── */
details > summary {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--ink) !important;
    font-weight: 600 !important; cursor: pointer;
    transition: border-color 0.2s;
}
details > summary:hover { border-color: var(--border3) !important; }
details[open] > summary { border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important; }
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--ink) !important;
    font-weight: 600 !important; transition: border-color 0.2s;
}
.streamlit-expanderHeader:hover { border-color: var(--border3) !important; }

/* ── Checkbox ── */
.stCheckbox label { color: var(--ink2) !important; font-size: 0.9rem !important; font-weight: 500 !important; }

/* ── Result Container ── */
.result-container {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-lg);
    padding: 2rem; margin-top: 1.5rem;
    box-shadow: var(--shadow-sm);
    animation: fadeUp 0.4s ease;
}
.result-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--violet); margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.result-content {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--violet);
    border-radius: var(--radius-md);
    padding: 1.5rem; color: var(--ink);
    line-height: 1.75; font-size: 0.92rem;
    white-space: pre-wrap; word-break: break-word;
    max-height: 560px; overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: var(--surface3) transparent;
}
.result-content::-webkit-scrollbar { width: 4px; }
.result-content::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem; background: var(--bg2);
    padding: 0.4rem; border-radius: var(--radius-md);
    margin-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    color: var(--ink2) !important;
    background: transparent !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ink) !important; background: var(--surface) !important; }
.stTabs [aria-selected="true"] {
    color: var(--violet) !important;
    background: var(--surface) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
}

/* ── Metrics ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.85rem; margin: 1rem 0;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--emerald);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    transition: border-left-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-left-color: var(--violet); transform: translateX(2px); }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: var(--violet);
}

/* ── Banners ── */
.banner {
    padding: 1rem 1.25rem; border-radius: var(--radius-md);
    border-left: 3px solid; margin: 0.75rem 0;
    font-size: 0.88rem; line-height: 1.6;
}
.banner.error  { background: var(--red-dim);    border-left-color: var(--red);     color: var(--red); }
.banner.warning{ background: var(--amber-dim);  border-left-color: var(--amber);   color: var(--amber); }
.banner.success{ background: var(--emerald-dim);border-left-color: var(--emerald); color: var(--emerald); }

/* ── Sidebar (Right) - RAG Knowledge Base ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-left: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
.sidebar-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin: 1.5rem 0 0.6rem;
}
.kb-title {
    font-size: 1.3rem; font-weight: 800;
    color: var(--ink); margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.kb-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid var(--border);
}
.kb-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
}
.kb-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem; font-weight: 700; color: var(--violet);
}
.kb-topic-tag {
    display: inline-block; padding: 0.3rem 0.65rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    color: var(--ink2); transition: all 0.2s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 100%; vertical-align: top;
}
.kb-topic-tag:hover { background: var(--violet-dim); border-color: var(--violet); color: var(--violet); }

/* ── Sidebar Button Override ── */
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    color: var(--ink2) !important;
    box-shadow: none !important;
    padding: 0.4rem 0.75rem !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--violet) !important;
    color: var(--violet) !important;
    transform: none !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--violet) !important; }

/* ── Hide File Uploader Text ── */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"] > div > div > span { display: none !important; }
.stFileUploader [data-testid="stMarkdownContainer"] { display: none !important; }
span.material-symbols-rounded { display: none !important; }

/* ── Divider ── */
.rule { border: none; height: 1px; background: var(--border); margin: 1.25rem 0; }

/* ── Danger zone ── */
.danger-btn .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(248,113,113,0.25) !important;
    color: var(--red) !important; box-shadow: none !important;
    font-weight: 600 !important;
}
.danger-btn .stButton > button:hover {
    background: var(--red-dim) !important; border-color: var(--red) !important;
}

/* ── Download ── */
.stDownloadButton > button {
    background: var(--surface) !important; color: var(--ink2) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important; transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--violet) !important; color: var(--violet) !important;
}

/* ── Alert ── */
.stAlert {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--violet) !important; }

/* ── Upload help ── */
.upload-help {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--muted); margin-top: 0.4rem;
}

/* ── Hide Streamlit elements ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton { display: none !important; }
[data-testid="stAppViewBlockContainer"] ~ *[class*="keyboard"],
div[role="tooltip"][class*="keyboard"],
/* Broad hide rules for stray icon ligatures and material icon names */
[class*="keyboard"],
[class^="keyboard"],
i[class*="keyboard"],
span[class*="keyboard"],
svg[class*="keyboard"],
.material-icons,
i.material-icons,
span.material-icons,
[aria-label*="keyboard"] { display: none !important; visibility: hidden !important; }

/* Specifically hide any tooltip or attribute-based labels inside the sidebar
   to prevent the 'keyboard_double_arrow' ligature text from showing on hover. */
[data-testid="stSidebar"] div[role="tooltip"],
[data-testid="stSidebar"] [title*="keyboard_double"],
[data-testid="stSidebar"] [aria-label*="keyboard_double"],
[data-testid="stSidebar"] [data-testid*="keyboard_double"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ── Fix file uploader ── */
[data-testid="stFileUploaderDropzone"] button { display: none !important; }
[data-testid="stFileUploaderDropzone"] small  { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { display: none !important; }
[data-testid="stFileUploaderDropzone"]:hover button { display: none !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

@media (max-width: 768px) {
    .app-title { font-size: 2.6rem; }
}
</style>
""", unsafe_allow_html=True)

# Remove stray 'keyboard_double...' text if it appears (runs repeatedly for dynamic elements)
st.markdown("""
<script>
function removeKeyboardDouble() {
    try {
        document.querySelectorAll('*').forEach(el => {
            try {
                if (el.childElementCount === 0 && el.innerText && el.innerText.toLowerCase().includes('keyboard_double')) {
                    el.remove();
                }
            } catch(e){}
        });
    } catch(e){}
}
// Run once and a few times after to catch dynamically inserted elements
removeKeyboardDouble();
setTimeout(removeKeyboardDouble, 500);
setTimeout(removeKeyboardDouble, 1500);
setInterval(removeKeyboardDouble, 5000);
</script>
""", unsafe_allow_html=True)

# Stronger remover: scan text nodes and hide ancestor elements and attributes containing the substring
st.markdown("""
<script>
;(function(){
    const substr = 'keyboard_double';
    function hideMatches(){
        try{
            // scan text nodes
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
            let node;
            const toHide = new Set();
            while(node = walker.nextNode()){
                try{
                    if(node.nodeValue && node.nodeValue.toLowerCase().includes(substr)){
                        let el = node.parentElement;
                        for(let i=0;i<6 && el;i++){
                            toHide.add(el);
                            el = el.parentElement;
                        }
                        node.nodeValue = '';
                    }
                }catch(e){}
            }
            toHide.forEach(el=>{
                try{
                    el.style.display = 'none' !important;
                    el.style.visibility = 'hidden' !important;
                    el.style.pointerEvents = 'none' !important;
                }catch(e){}
            });

            // hide by attributes (title, aria-label, data-testid)
            const attrSelector = ['title','aria-label','data-testid'];
            attrSelector.forEach(attr=>{
                document.querySelectorAll('['+attr+'*="'+substr+'"]').forEach(el=>{
                    try{ el.style.display='none'; el.style.visibility='hidden'; el.style.pointerEvents='none'; }catch(e){}
                });
            });
        }catch(e){}
    }
    hideMatches();
    setTimeout(hideMatches,200);
    setTimeout(hideMatches,800);
    const id = setInterval(hideMatches,3000);
    // stop after a while to avoid overhead
    setTimeout(()=>clearInterval(id), 60000);
})();
</script>
""", unsafe_allow_html=True)


# (Sidebar forced position removed - RAG sidebar now works normally on the right)
import uuid

# ── Initialize Session State ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "research_results" not in st.session_state:
    st.session_state.research_results = []
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "pipeline_visible" not in st.session_state:
    st.session_state.pipeline_visible = True



# ── Error Page ────────────────────────────────────────────────────────────────
if not BACKEND_OK:
    st.markdown(f"""
    <div class="banner error">
        <strong>⚠️ Backend Import Error</strong><br>
        {BACKEND_ERR}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <div class="app-eyebrow">Multi-Agent AI System</div>
    <h1 class="app-title">
        <span class="brand">Research</span><span class="brand-accent">Mind</span>
    </h1>
    <p class="app-subtitle">
        Four specialized AI agents collaborate – searching, scraping, writing, and critiquing – to deliver a polished research report on any topic.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Interface
st.markdown('<div class="section-header">Research Topic</div>', unsafe_allow_html=True)

topic_input = st.text_input(
    "Enter your research topic",
    placeholder="e.g., Quantum computing breakthroughs in 2025",
    label_visibility="collapsed",
    key="topic_input",
)

# Advanced Options
with st.expander("⚙️ Advanced Options"):
    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        use_rag = st.checkbox(
            "Enable RAG (Retrieval Augmented Generation)",
            value=True,
            help="Use knowledge base for context-aware responses"
        )

        run_ragas = st.checkbox(
            "Run RAGAS Evaluation",
            value=False,
            help="Evaluate report quality with RAGAS metrics"
        )

    with col_opt2:
        comparison_mode = st.checkbox(
            "Compare RAG vs Plain LLM",
            value=False,
            help="Generate both RAG-augmented and plain reports"
        )

# Run Button
run_clicked = st.button("▶ Run Research Pipeline", use_container_width=True, type="primary")

# ── Agent Flow Pipeline Section ────────────────────────────────────────────────
if st.session_state.pipeline_visible:
    st.markdown('<div class="section-header" style="margin-top: 2.5rem;">Pipeline Stages</div>', unsafe_allow_html=True)
    st.markdown('<div class="kb-title" style="margin-bottom: 1.5rem;">Agent Flow</div>', unsafe_allow_html=True)
    
    pipeline_stages = [
        {"number": "01", "title": "Search Agent", "desc": "Gathers latest web information", "status": "ready"},
        {"number": "02", "title": "Reader Agent", "desc": "Scrapes & extracts deep content", "status": "ready"},
        {"number": "03", "title": "Writer Chain", "desc": "Crafts structured report", "status": "ready"},
        {"number": "04", "title": "Critic Chain", "desc": "Reviews & provides feedback", "status": "ready"}
    ]

    cols = st.columns(4)
    for idx, stage in enumerate(pipeline_stages):
        with cols[idx]:
            status_class = stage["status"]
            status_icon = "✔" if status_class == "ready" else "◌"
            st.markdown(f"""
            <div class="pipeline-card">
                <div class="pipeline-number">{stage["number"]}</div>
                <div class="pipeline-title">{stage["title"]}</div>
                <div class="pipeline-desc">{stage["desc"]}</div>
                <div class="pipeline-status {status_class}">{status_icon} {status_class.upper()}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def _extract(content):
    """Extract text from AIMessage or return as-is."""
    if hasattr(content, "content"):
        return content.content
    return str(content)


def _extract_markdown(content):
    """Extract markdown report from <report>...</report> tags."""
    text = _extract(content)
    match = re.search(r"<report>(.*?)</report>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_json(content):
    """Extract JSON from <json>...</json> tags."""
    text = _extract(content)
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    return match.group(1).strip() if match else text


# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_clicked:
    if not topic_input.strip():
        st.warning("⚠️ Please enter a research topic first.")
    else:
        topic = topic_input.strip()
        st.session_state.active_topic = topic

        result = {
            "topic": topic,
            "search_results": None,
            "scraped_content": None,
            "rag_context": "",
            "rag_accepted": [],
            "rag_rejected": [],
            "rag_used": False,
            "report": None,
            "critique": None,
            "comparison_mode": comparison_mode,
            "report_plain": None,
            "ragas_eval": {},
        }

        scrape_ok = False

        try:
            # ── Stage 1: Search Agent ─────────────────────────────────────────
            with st.status("🔍 Search Agent: Gathering information...", expanded=True) as status:
                st.write("Querying web search...")
                search_agent = build_search_agent()
                search_result = search_agent.invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
                })
                search_text = _extract(search_result["messages"][-1])
                result["search_results"] = search_text
                st.write("✅ Search complete")
                status.update(label="✅ Search Agent Complete", state="complete")

            # ── Stage 2: Reader Agent ─────────────────────────────────────────
            with st.status("📖 Reader Agent: Extracting content...", expanded=True) as status:
                st.write("Scraping top source...")
                reader_agent = build_reader_agent()
                reader_result = reader_agent.invoke({
                    "messages": [("user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{search_text[:800]}")]
                })
                scraped_text = _extract(reader_result["messages"][-1])
                result["scraped_content"] = scraped_text
                scrape_ok = len(scraped_text.strip()) >= 120
                st.write("✅ Content extracted")
                status.update(label="✅ Reader Agent Complete", state="complete")

            # ── Stage 3: RAG Engine ───────────────────────────────────────────
            rag_context_str = ""
            with st.status("🧠 RAG Engine: Retrieving knowledge...", expanded=True) as status:
                rag_context_str, rag_accepted, rag_rejected = retrieve_relevant_context(
                    topic,
                    k=5,
                    exclude_session_id=st.session_state.session_id,
                )
                result["rag_context"] = rag_context_str
                result["rag_accepted"] = rag_accepted
                result["rag_rejected"] = rag_rejected
                result["rag_used"] = len(rag_accepted) > 0

                source_urls = re.findall(r'https?://[^\s\)\"\'>]+', search_text)
                add_research_to_kb(
                    topic, search_text, source_urls,
                    source_type="web_search",
                    session_id=st.session_state.session_id,
                )
                if scrape_ok:
                    add_research_to_kb(
                        topic, scraped_text, source_urls,
                        source_type="web_scrape",
                        session_id=st.session_state.session_id,
                    )

                chunks_found = len(rag_accepted)
                st.write(f"✅ {chunks_found} relevant chunks retrieved from knowledge base")
                status.update(label=f"✅ RAG: {chunks_found} chunks found", state="complete")

            # ── Stage 4: Writer Chain ─────────────────────────────────────────
            with st.status("✍️ Writer Chain: Crafting report...", expanded=True) as status:
                st.write("Generating research report...")
                research_combined = (
                    f"SEARCH RESULTS:\n{search_text}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n"
                    f"{scraped_text if scrape_ok else 'Scrape unavailable – using search snippets only.'}"
                )
                no_rag_msg = "No relevant past research available for this topic."
                report = writer_chain.invoke({
                    "topic": topic,
                    "research": research_combined,
                    "rag_context": rag_context_str if result["rag_used"] else no_rag_msg,
                })
                result["report"] = report
                st.write("✅ Report generated")
                status.update(label="✅ Writer Chain Complete", state="complete")

            # ── Comparison Mode ───────────────────────────────────────────────
            if comparison_mode and result["rag_used"]:
                with st.status("🔄 Comparison: Plain LLM (no RAG)...", expanded=True) as status:
                    report_plain = writer_chain.invoke({
                        "topic": topic,
                        "research": research_combined,
                        "rag_context": no_rag_msg,
                    })
                    result["report_plain"] = report_plain
                    status.update(label="✅ Comparison Complete", state="complete")

            # ── Stage 5: Critic Chain ─────────────────────────────────────────
            with st.status("🎯 Critic Chain: Reviewing report...", expanded=True) as status:
                st.write("Analysing report quality...")
                critique = critic_chain.invoke({"report": report})
                result["critique"] = critique
                st.write("✅ Review complete")
                status.update(label="✅ Critic Chain Complete", state="complete")

            # ── Stage 6: RAGAS Evaluation ─────────────────────────────────────
            if run_ragas and result["rag_used"]:
                with st.status("📊 RAGAS Evaluation...", expanded=True) as status:
                    try:
                        eval_contexts = build_eval_contexts(
                            search_text, scraped_text, rag_context_str
                        )
                        ragas_result = evaluate_report_with_ragas(
                            question=topic,
                            answer=report,
                            contexts=eval_contexts,
                        )
                        result["ragas_eval"] = ragas_result
                        st.write("✅ RAGAS evaluation complete")
                        status.update(label="✅ RAGAS Complete", state="complete")
                    except Exception as e:
                        result["ragas_eval"] = {"ok": False, "skipped": True, "error": str(e), "scores": {}, "overall": None}
                        status.update(label="⚠️ RAGAS Failed", state="error")
            else:
                result["ragas_eval"] = {
                    "ok": False, "skipped": True,
                    "error": "RAG not used – RAGAS skipped to avoid misleading scores.",
                    "scores": {}, "overall": None,
                }

            st.session_state.research_results.append(result)
            st.success("✨ Pipeline complete! Scroll down for results.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")


# ── Display Results ────────────────────────────────────────────────────────────
if st.session_state.research_results:
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    for idx, r in enumerate(reversed(st.session_state.research_results)):
        with st.container():
            num = len(st.session_state.research_results) - idx
            st.markdown(
                f'<div class="result-container">'
                f'<div class="result-header">Report #{num} &mdash; {r.get("topic", "")}</div>',
                unsafe_allow_html=True,
            )

            tab_names = ["Report", "Search Results", "Scraped Content", "RAG Context", "Critique", "RAGAS Eval"]
            if r.get("comparison_mode") and r.get("report_plain"):
                tab_names.append("Comparison")
            tabs = st.tabs(tab_names)

            with tabs[0]:
                rpt = r.get("report") or ""
                if rpt.strip():
                    st.markdown(rpt)
                else:
                    st.info("No report generated.")

            with tabs[1]:
                st.markdown(f'<div class="result-content">{r.get("search_results") or ""}</div>', unsafe_allow_html=True)

            with tabs[2]:
                st.markdown(f'<div class="result-content">{r.get("scraped_content") or ""}</div>', unsafe_allow_html=True)

            with tabs[3]:
                accepted = r.get("rag_accepted", [])
                rejected = r.get("rag_rejected", [])
                st.markdown(
                    f'<div class="metric-grid">'
                    f'<div class="metric-card"><div class="metric-label">Accepted Chunks</div><div class="metric-value">{len(accepted)}</div></div>'
                    f'<div class="metric-card"><div class="metric-label">Filtered Out</div><div class="metric-value">{len(rejected)}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if accepted:
                    for i, chunk in enumerate(accepted, 1):
                        with st.expander(f"Chunk {i} | sim {chunk.get('similarity', '?')} | {chunk.get('source_type', '?')}"):
                            st.write(chunk.get("preview", ""))
                else:
                    rag_str = r.get("rag_context", "")
                    if rag_str:
                        st.markdown(f'<div class="result-content">{rag_str}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No prior knowledge retrieved for this topic.")

            with tabs[4]:
                critique = r.get("critique") or "No critique generated."
                st.markdown(f'<div class="result-content">{critique}</div>', unsafe_allow_html=True)

            with tabs[5]:
                ragas_eval = r.get("ragas_eval", {})
                if ragas_eval.get("skipped"):
                    st.info(ragas_eval.get("error", "RAGAS was skipped."))
                elif ragas_eval.get("ok"):
                    scores = ragas_eval.get("scores", {})
                    overall = ragas_eval.get("overall", 0)
                    score_html = "".join(
                        f'<div class="metric-card"><div class="metric-label">{k.replace("_", " ")}</div><div class="metric-value">{v:.3f}</div></div>'
                        for k, v in scores.items()
                    )
                    st.markdown(
                        f'<div class="metric-card" style="border-left-color:var(--violet);margin-bottom:1rem;">'
                        f'<div class="metric-label">Overall Score</div><div class="metric-value">{overall:.3f}</div></div>'
                        f'<div class="metric-grid">{score_html}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(ragas_eval.get("error", "RAGAS not run."))

            if r.get("comparison_mode") and r.get("report_plain") and len(tabs) > 6:
                with tabs[6]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**With RAG**")
                        st.markdown(r.get("report", ""))
                    with col2:
                        st.markdown("**Plain LLM (no RAG)**")
                        st.markdown(r.get("report_plain", ""))

            dl_text = r.get("report") or ""
            safe_t = r.get("topic", "report")[:40].replace(" ", "_")
            st.download_button(
                "Download Report",
                data=dl_text,
                file_name=f"report_{safe_t}.txt",
                mime="text/plain",
                key=f"dl_{idx}",
            )
            st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar: Knowledge Base (Right) ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">Knowledge Base</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="kb-title">RAG Memory</div>', unsafe_allow_html=True)
    
    try:
        kb = get_kb_stats()
    except Exception as e:
        st.error(f"KB Error: {str(e)[:100]}")
        kb = {"exists": False, "total_chunks": 0, "topics": [], "source_breakdown": {}, "papers": []}
    
    breakdown = kb.get("source_breakdown", {"web_search": 0, "web_scrape": 0, "paper": 0})
    st.markdown(
        f'<div class="kb-stat"><span class="kb-stat-label">Chunks indexed</span><span class="kb-stat-value">{kb["total_chunks"]}</span></div>'
        f'<div class="kb-stat"><span class="kb-stat-label">Topics stored</span><span class="kb-stat-value">{len(kb["topics"])}</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin-top:1rem;">'
        f'<div style="padding:0.75rem;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);text-align:center;"><div style="font-family:JetBrains Mono;font-size:0.6rem;color:var(--muted);text-transform:uppercase;">Search</div><div style="font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;color:var(--ink);">{breakdown.get("web_search", 0)}</div></div>'
        f'<div style="padding:0.75rem;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);text-align:center;"><div style="font-family:JetBrains Mono;font-size:0.6rem;color:var(--muted);text-transform:uppercase;">Scrape</div><div style="font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;color:var(--ink);">{breakdown.get("web_scrape", 0)}</div></div>'
        f'<div style="padding:0.75rem;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-sm);text-align:center;"><div style="font-family:JetBrains Mono;font-size:0.6rem;color:var(--violet);text-transform:uppercase;">Papers</div><div style="font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;color:var(--violet);">{breakdown.get("paper", 0)}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    topics = dedupe_topics(kb.get("topics", []))
    if topics:
        st.markdown('<div style="margin-top:1.25rem;"><span class="kb-stat-label">Recent Topics</span></div>', unsafe_allow_html=True)
        tags_html = "".join(
            f'<span class="kb-topic-tag">{t[:30]}{"..." if len(t) > 30 else ""}</span>'
            for t in topics[-10:]
        )
        st.markdown(f'<div style="margin-top:0.5rem;">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Upload Paper</div>', unsafe_allow_html=True)
    
    try:
        uploaded_pdf = st.file_uploader(
            "Upload a PDF research paper",
            type=["pdf"],
            key="paper_uploader",
            label_visibility="collapsed",
        )
        st.markdown('<div class="upload-help">PDF only &middot; 200MB max</div>', unsafe_allow_html=True)

        if uploaded_pdf is not None:
            if st.button("Ingest Paper", use_container_width=True):
                with st.spinner("Extracting and embedding paper..."):
                    pdf_result = add_paper_to_kb(
                        file_name=uploaded_pdf.name,
                        file_bytes=uploaded_pdf.read(),
                        session_id=st.session_state.get("session_id", "upload"),
                    )
                if pdf_result["success"]:
                    st.success(f"Ingested {uploaded_pdf.name} - {pdf_result['pages']} pages, {pdf_result['chunks']} chunks")
                    st.rerun()
                else:
                    st.error(pdf_result["error"])
    except Exception as e:
        st.warning(f"Paper upload temporarily unavailable: {str(e)[:80]}")

    papers = kb.get("papers", [])
    if papers:
        st.markdown('<div style="margin-top:1rem;"><span class="kb-stat-label">Ingested Papers</span></div>', unsafe_allow_html=True)
        for p in papers[-6:]:
            try:
                nm = p["name"][:35] + ("..." if len(p["name"]) > 35 else "")
                st.markdown(
                    f'<div style="padding:0.6rem 0.875rem;margin:0.4rem 0;background:var(--surface);border:1px solid var(--border2);border-left:2px solid var(--violet);border-radius:var(--radius-sm);">'
                    f'<div style="font-family:JetBrains Mono;font-size:0.7rem;font-weight:600;color:var(--ink2);">{nm}</div>'
                    f'<div style="font-family:JetBrains Mono;font-size:0.6rem;color:var(--muted);">{p["pages"]} pages &middot; {p["chunks"]} chunks</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                continue

    if kb["exists"]:
        st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        try:
            if st.button("Clear Knowledge Base", use_container_width=True):
                clear_kb()
                st.rerun()
        except Exception as e:
            st.error(f"Clear KB error: {str(e)[:80]}")
        st.markdown('</div>', unsafe_allow_html=True)
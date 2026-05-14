# 🚀 Multi-Agent AI Research System

> A production-style multi-agent pipeline that performs end-to-end research using web data, Retrieval-Augmented Generation (RAG), and quantitative evaluation with RAGAS.

---

## ✨ Overview

This project simulates a **collaborative AI research system** where multiple specialized agents work together to:

* 🔍 Search real-time information
* 🌐 Scrape and extract structured content
* 🧠 Retrieve context using FAISS-based RAG
* ✍️ Generate high-quality research reports
* 🧑‍⚖️ Critically evaluate outputs
* 📊 Measure quality using RAGAS

---

## 🧠 Architecture

```mermaid
flowchart TD

A[User Query]

A --> B[01 — Search Agent<br/>(Tavily Web Search)]
B --> C[02 — Reader Agent<br/>(Scrape & Extract Content)]

C --> D[03 — RAG Engine<br/>(Chunk + Embed + Retrieve)]

D -->|Persist| E[(FAISS Vector DB<br/>Knowledge Base)]
E -->|Retrieve| D

D --> F[04 — Writer Agent<br/>(Generate Research Report)]
F --> G[05 — Critic Agent<br/>(Review & Improve Output)]

G --> H[Final Report + RAGAS Evaluation]
```

---

## ⚙️ Features

* 🤖 Multi-agent orchestration (Search → Read → Write → Critique)
* 📚 RAG with FAISS for semantic retrieval
* 🌍 Web + PDF document ingestion
* 📊 RAGAS evaluation (Faithfulness + Answer Relevancy)
* ⚡ Interactive Streamlit UI
* 🔁 RAG vs Plain LLM comparison mode
* 🧠 Context selection with similarity scoring

---

## 📊 Example Performance

| Scenario               | Faithfulness | Relevancy |
| ---------------------- | ------------ | --------- |
| Research Paper (BERT)  | 0.98         | 0.81      |
| Web Query (Oil Prices) | 0.87         | 0.91      |
| Broad Topic (Economy)  | 0.30–0.70    | ~0.80     |

👉 Demonstrates how retrieval quality directly impacts grounding and hallucination reduction.

---

## 📸 Demo

> *(Add your screenshots here for maximum impact)*
![UI](image.png)
![Working](<Screenshot 2026-05-05 035324.png>)
![RAGAS Evaluation](image-1.png)


```md

```

---

## 🛠️ Tech Stack

* **Language**: Python
* **Frameworks**: LangChain
* **Vector DB**: FAISS
* **Search API**: Tavily
* **Scraping**: BeautifulSoup
* **UI**: Streamlit
* **Evaluation**: RAGAS

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/multi-ai-research-agent.git
cd multi-ai-research-agent

pip install -r requirements.txt
streamlit run app1.py
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
TAVILY_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

---

## 📦 Notes

* FAISS index files are not included (generated at runtime)
* Designed for experimentation with RAG pipelines and evaluation

---

## 🚀 Future Improvements

* FastAPI + Next.js full-stack deployment
* Smarter chunk deduplication
* Hybrid retrieval (BM25 + embeddings)
* Multi-modal support (images, tables)
* Agent memory optimization

---

## 💡 Key Insight

This project demonstrates that:

> **Combining multi-agent systems with RAG + evaluation loops significantly improves factual accuracy and reduces hallucinations in LLM outputs.**

---

## 🧑‍💻 Author

**Shaurya Singh**

---

## ⭐ If you found this useful, consider starring the repo!

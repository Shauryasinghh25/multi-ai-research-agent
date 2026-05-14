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

A --> B[01 - Search Agent (Tavily Web Search)]
B --> C[02 - Reader Agent (Scrape & Extract Content)]

C --> D[03 - RAG Engine (Chunk + Embed + Retrieve)]

D -->|Persist| E[(FAISS Vector DB - Knowledge Base)]
E -->|Retrieve| D

D --> F[04 - Writer Agent (Generate Research Report)]
F --> G[05 - Critic Agent (Review & Improve Output)]

G --> H[Final Report + RAGAS Evaluation]
```

---

## ⚙️ Features

* 🤖 Multi-agent orchestration (Search → Read → Write → Critique)
* 📚 RAG with FAISS for semantic retrieval
* 🌍 Web + PDF document ingestion
* 📊 RAGAS evaluation (Faithfulness & Answer Relevancy)
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

👉 Shows how retrieval quality impacts factual grounding.

---

## 📸 Demo

> Add your screenshots inside an `assets/` folder for clean rendering.

```bash
assets/
├── ui.png
├── pipeline.png
├── ragas.png
```

```md
![UI](assets/ui.png)
![Pipeline](assets/pipeline.png)
![RAGAS Evaluation](assets/ragas.png)
```

---

## 🛠️ Tech Stack

* **Language**: Python
* **Framework**: LangChain
* **Vector Database**: FAISS
* **Search API**: Tavily
* **Scraping**: BeautifulSoup
* **UI**: Streamlit
* **Evaluation**: RAGAS

---

## ▶️ Run Locally

```bash
git clone https://github.com/Shauryasinghh25/multi-ai-research-agent.git
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

* FAISS index files are excluded (generated at runtime)
* Designed for experimentation with RAG pipelines and evaluation

---

## 🚀 Future Improvements

* FastAPI + Next.js deployment
* Improved chunk deduplication
* Hybrid retrieval (BM25 + embeddings)
* Multi-modal support
* Agent memory optimization

---

## 💡 Key Insight

> Combining multi-agent systems with RAG + evaluation loops significantly improves factual accuracy and reduces hallucinations in LLM outputs.

---

## 🧑‍💻 Author

**Shaurya Singh**

---

## ⭐ If you found this useful, consider starring the repo!

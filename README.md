# SHL Conversational Assessment Recommender

An AI-powered conversational agent that recommends SHL Individual Test Solutions based on natural-language hiring requirements.

---

## 🎨 UI/UX Highlights
- **Premium Glassmorphic Design**: Frosted panels, subtle warm/cool gradients, and dynamic glowing indicators.
- **Embedded Table Rendering**: Integrates a custom Markdown table parser to display assessment shortlists in neat, responsive HTML tables.
- **Color-Coded Badges**: Dynamic status chips for test types (e.g. `P` -> `Personality` in Emerald, `A` -> `Ability` in Blue).
- **Smooth Interaction**: Shimmering typing bubbles and custom scrollbars for executive-level presentation.

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env
# Edit .env and paste your OPENROUTER_API_KEY

# 3. Build retrieval index (Enriches metadata and creates vectors)
uv run python scripts/build_index.py

# 4. Start the API & UI server
./run.sh
```

---

## 🛠 Architecture & Retrieval Design

```
User → POST /chat (full messages list)
         │
         ▼
    Guardrails (regex scope checks & injections)
         │
         ▼
    Fact Extraction (Structured LLM extracts role, seniority, constraints)
         │
         ▼
    Intent Classification & Context Gate (Enforces Turn 1 & 2 clarification)
         │
    ┌────┴───────────────────────────────┐
    │ clarify │ recommend │ compare │ closing │
    └────┬───────────────────────────────┘
         │
    Richer RAG Formulation & Hybrid Retrieval (Dense ChromaDB + BM25 + RRF)
         │
    Post-Retrieval Job-level Filter (Grounded matching)
         │
    Response Generation (Formatted tabular Markdown reply)
         │
    ChatResponse → User (Renders in Web Client)
```

### Key Core Features:
1. **Context-guessing Prevention**: The agent will not guess if you're assessing for *selection* or *development*. It will cleanly prompt a clarifying question on Turn 1 to build a accurate shortlist.
2. **Hybrid Search (Dense + Sparse)**: Reciprocal Rank Fusion (RRF) combines cosine vector similarities with BM25 keyword matching (perfect for exact terms like "Java" or "Spring").
3. **Domain Filtering**: Prevents misaligned recommendations (e.g. keeps Sales Interview Guides away from general leadership selection shortlists).

---

## 🚀 Deploying to Vercel

The project is fully pre-configured for Vercel Serverless deployments.

### 1. Build and Commit Search Indices
Make sure your indices are built locally so they are bundled with the Vercel function:
```bash
uv run python scripts/build_index.py
```

### 2. Deploy via Vercel CLI
```bash
# Login to Vercel
vercel login

# Run deployment
vercel

# Add environment variables
vercel env add OPENROUTER_API_KEY
# (Set OPENROUTER_API_KEY, LLM_MODEL, EMBEDDING_MODEL)

# Deploy to production
vercel --prod
```

---

## 🧪 Running Tests

```bash
# Run pytest test suite
uv run pytest tests/ -v

# Run behavior probes (Requires API key)
uv run pytest eval/probes.py -v

# Run trace replay harness
uv run python scripts/run_eval.py
```

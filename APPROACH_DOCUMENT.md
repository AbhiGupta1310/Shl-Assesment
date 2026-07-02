# How I Built the SHL Assessment Recommender

Hi! Here's a quick overview of how I designed and built this conversational agent. I wanted to make sure it was robust against hallucinations while still feeling natural to chat with. 

## 1. Core Design Choices

- **Stateless API:** Instead of dealing with messy server-side session databases, I just pass the full conversation history on every API call. It means the backend has to re-extract facts on every turn, but for normal conversation lengths (under 20 turns), it’s super fast and makes the app completely reproducible and easy to host on Vercel.
- **Strict Readiness Gate:** Early on, the bot would awkwardly try to recommend tests when a user just said "Hi, I need a test." I built an explicit Python gate (`is_ready_to_recommend`) that blocks recommendations until the bot actually knows the role/skill AND either the seniority, context, or a direct request.
- **Anti-Hallucination Checks:** LLMs love inventing URLs. To fix this, I made a strict rule in the code: every URL the bot tries to output is validated against `catalog.json`. If it's a hallucinated link, it silently gets dropped before the user ever sees it.

## 2. Retrieval Setup (What worked & what didn't)

- **What didn't work:** At first, I tried pure vector search (embeddings). It was good at understanding concepts, but it completely failed when someone asked for a specific technical skill like "Java" (it would return generic coding tests). Then I tried pure keyword search (BM25), but it totally missed semantic matches (e.g., someone asking for "interpersonal skills" wouldn't match "stakeholder communication").
- **The Solution:** I built a **Hybrid Search** pipeline. I used `openai/text-embedding-3-small` for dense embeddings and combined it with BM25 using Reciprocal Rank Fusion (RRF). I weighted it 60% dense and 40% sparse. Now, it catches both exact keywords and conceptual meaning!

## 3. Prompt Design

Instead of one massive, confusing "god prompt", I split the bot's brain into a few smaller, focused prompts:
- **Intent Classification:** A quick LLM pass that decides if the user is refining, ready for a recommendation, or just chatting.
- **Structured Fact Extraction:** Instead of using clunky regex to find out if they want "remote testing" or a "senior" role, I prompt the LLM to read the whole chat history and output a strict JSON of known facts. It’s way better at handling messy human conversation.
- **Strict Grounding:** The actual recommendation prompt is forced to *only* choose from the candidates passed to it by the retrieval engine and format it into a clean Markdown table.

## 4. Evaluation Approach & Measuring Improvement

To actually know if my tweaks were making the bot better, I built a custom evaluation harness (`eval/harness.py`). 
- **The Setup:** I saved a bunch of real conversation traces. The harness replays these chats turn-by-turn without needing the server running.
- **The Metric:** I measured improvement using **Recall@10**. The harness checks if the expected, normalized SHL product URLs are actually present in the final recommendation list. 
- Being able to run this test suite meant I could tweak the RRF weights (like moving dense from 0.5 to 0.6) or edit a prompt and immediately see if the recall score improved.

## 5. AI Tools Used

I definitely got some help from AI tools to build this faster:
- **Claude Opus:** I used Claude Opus heavily to write basically the entire frontend. I wanted a really premium, glassmorphic UI with dynamic tables and status badges, and Opus nailed the HTML/CSS. I also used it to brainstorm and experiment with the hybrid retrieval logic.
- **Agentic Coding:** I used agentic AI coding tools (like this one!) to help scaffold the Python backend, write the boilerplate for the FastAPI/Vercel setup, and help write the evaluation harness scripts. It saved me a ton of time on debugging. 

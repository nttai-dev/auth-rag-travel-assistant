# Design Document: Lumen Travel — Cần Thơ Sign-In Travel Assistant with RAG Chat

**Date:** 2026-11-13  
**Course / Assignment:** Computational Thinking — Practice 03 (Sign-In Travel Assistant with RAG Chat)  
**Target City:** Cần Thơ, Vietnam

---

## 1. Intent and the Why

*Lumen Travel*, a student start-up, needs a trustworthy travel assistant web app for **Cần Thơ**. General-purpose chatbots frequently invent opening hours, ticket prices, and local transport details. To prevent hallucination, this Streamlit assistant must:
1. Restrict access so **only signed-in users** (via Firebase Email/Password Authentication from Lab 3) can view and use the chat page.
2. Answer questions **strictly from a custom travel guide** (`knowledge_base.json`, 12–20 entries written in our own words), citing entry IDs such as `[G01]` and showing the retrieved entries and their cosine similarity scores in an expandable UI panel.
3. Explicitly reply **`"I don't know"`** when the travel guide does not contain the answer instead of guessing.
4. Serve as the clean architectural foundation for **Assignment 4 (RAG-Rec)** by including structured `category` and `tags` fields on place entries to support the `filter → score → top-k` recommendation pattern.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Firebase Auth Protection:** Provide email/password sign-up, sign-in, and sign-out via Firebase Authentication REST APIs (`firebase_client.py`), hiding the RAG chat interface until signed in.
- **Session-Only Chat History:** Store conversation history in `st.session_state["chat_history"]` and clear all user/chat state immediately upon sign-out (`forget_user()`).
- **Curated Cần Thơ Knowledge Base:** Create `knowledge_base.json` with **16 entries (`G01`–`G16`)** in English written in our own words (`id`, `title`, `text`, `source`, plus `category` and `tags` for place entries).
- **Single-Time Guide Embedding:** Embed all guide entries in one batch API call to Ollama (`embeddinggemma`) cached via `@st.cache_resource` so document embeddings are computed once per server lifecycle, not per question.
- **Two-Layer Grounded RAG (`rag.py`):**
  - Retrieve `top_k = 3` entries by cosine similarity.
  - Apply a retrieval similarity threshold (`SIMILARITY_THRESHOLD = 0.35`) and a strict generation prompt (`qwen3:4b`) so out-of-scope questions reliably produce `"I don't know"`.
  - Cite source IDs (e.g. `[G01]`) in answers and render retrieved entries + similarity scores inside an `st.expander` panel.
- **Clear Error Handling:** Gracefully handle and display user-friendly messages for missing `.env` config, wrong sign-in credentials, expired sessions, empty questions, and unreachable Ollama/Pinggy servers.
- **Complete Deliverables:** Provide `README.md` with setup instructions, the required **6-question test table** (3 answerable, 1 paraphrase, 2 not in the guide, with failure-step diagnosis), known limitations, AI tool usage note, and a structured **Computational Thinking** reference for the 5-minute demonstration video.

### Non-Goals
- **No Database / No Firestore:** Per specification (*"Do not use Firestore or any other database"*), chat history and user data are never written to Firestore or local disk.
- **No OAuth / Google Sign-In:** Only email/password authentication is needed.
- **No Recommendation UI Yet:** Place recommendation logic belongs to Assignment 4 (RAG-Rec), though `knowledge_base.json` includes `category` and `tags` now.

### Anti-Goals (Failure Conditions Even If Code Runs)
- Guessing or using pretrained world knowledge about Cần Thơ when a detail (e.g. a phone number, hotel price, or flight schedule) is absent from `knowledge_base.json`.
- Calling `/api/embed` for the full travel guide on every question instead of caching with `@st.cache_resource`.
- Hardcoding `FIREBASE_API_KEY` or `OLLAMA_BASE_URL` in Python files, or committing `.env` to Git.
- Crashing with raw Python tracebacks or leaking API keys/URLs when Pinggy or Firebase is unreachable.

---

## 3. Constraints

- **Runtime & Libraries:** Python 3.11+, `streamlit`, `requests`, `python-dotenv`, `numpy`.
- **External Services:**
  - Firebase Authentication REST API (`https://identitytoolkit.googleapis.com/v1/accounts:signUp` and `signInWithPassword`).
  - Lab 4 Ollama server exposed over a Pinggy HTTPS tunnel (`OLLAMA_BASE_URL` in `.env`), providing `embeddinggemma` (`/api/embed`) and `qwen3:4b` (`/api/chat`).
- **Repository Structure at Root:**
  - `app.py` — Streamlit UI (auth gate, chat history, input validation, expandable source viewer).
  - `rag.py` — Knowledge base loader, batch embedding (`embeddinggemma`), cosine similarity search (`top_k = 3`), and grounded answer generation (`qwen3:4b`).
  - `firebase_client.py` — Firebase REST Auth client adapted from Lab 3 (Firestore functions removed).
  - `config.py` — Loads `.env` variables (`FIREBASE_PROJECT_ID`, `FIREBASE_API_KEY`, `OLLAMA_BASE_URL`, `EMBED_MODEL`, `CHAT_MODEL`).
  - `knowledge_base.json` — 16 Cần Thơ guide entries.
  - `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.

---

## 4. Approaches Considered

### Approach 1 (Selected): Two-Layer Guardrail (Cosine Threshold + Strict System Prompt) & 4-Module Architecture
- **Description:** Splits code into `config.py`, `firebase_client.py`, `rag.py`, and `app.py`. Uses `@st.cache_resource` to embed `title + category + tags + text` of all 16 entries once. On each question, computes cosine similarity for `top_k = 3`.
  - **Layer 1 (Retrieval Guardrail):** If `max(scores) < 0.35`, skips LLM generation and immediately returns `"I don't know based on the travel guide."` (while still returning the top-3 entries and scores for inspection).
  - **Layer 2 (Generation Guardrail):** If `max(scores) >= 0.35`, calls `qwen3:4b` (`"think": False`) with a strict system prompt requiring `[Gxx]` citations and instructing the model to reply `"I don't know based on the travel guide."` if the retrieved entries do not answer the specific question.
- **Trade-offs:** Fast rejection of completely unrelated queries, strong protection against 4B model hallucination, clear separation between *retrieval* failure and *generation* failure for the 6-question test table, and a concrete algorithmic rule to explain in the demo video. Requires a moderate threshold (`0.35`) so paraphrased queries are not prematurely filtered out.

### Approach 2: Prompt-Only Guardrail (No Similarity Threshold)
- **Description:** Always sends the top-3 retrieved entries to `qwen3:4b` regardless of similarity score and relies solely on the system prompt to say `"I don't know"`.
- **Trade-offs:** Slightly simpler code, but adds unnecessary Pinggy/LLM latency on obviously irrelevant queries and increases the risk that a 4B model fabricates connections from low-relevance context.

### Approach 3: High Threshold-Only Guardrail & Single-File App
- **Description:** Uses a strict similarity cutoff (e.g. `0.55`) to decide answerability and keeps RAG functions inside `app.py`.
- **Trade-offs:** Rejects valid paraphrase queries whose cosine similarity typically lands around `0.40–0.52`, cannot detect unanswerable questions that mention a real Cần Thơ landmark (high similarity score) but ask for a missing attribute, and makes Assignment 4 harder to extend.

**Why Approach 1 was chosen:** It directly supports the assignment's rubric for both the 6-question error analysis table (distinguishing *missing data*, *retrieval*, and *generation*) and the Computational Thinking video (explaining `k` and the `"I don't know"` rule with a concrete example).

---

## 5. Detailed Design Decisions

### 5.1 Decisions Made with User
1. **City Selection:** **Cần Thơ** is the subject of the travel guide.
2. **Reuse of Labs 3 & 4:** Directly inherit Lab 3's `firebase_client.py` REST Auth flow + `forget_user()` session pattern (removing Firestore note calls) and Lab 4's `/api/embed` + NumPy cosine similarity pipeline.
3. **Architecture & Guardrail:** Implement **Approach 1** (Two-layer `"I don't know"` guardrail + 4 modular Python files at repo root).

### 5.2 Builder Decisions (Defaults Set by Builder)
1. **English Guide & Prompts (Builder Call):** All 16 entries in `knowledge_base.json` and the 6 test questions in `README.md` are written in English so `embeddinggemma`, `qwen3:4b`, category/tag tokens, and the required `"I don't know"` phrase work without cross-lingual drift.
2. **Knowledge Base Composition (`knowledge_base.json`, 16 Entries) (Builder Call):**
   - **12 Place Entries (`G01`–`G12`)** with `id`, `title`, `text`, `source`, `category`, and `tags` (for Assignment 4 RAG-Rec):
     - `G01`: Cái Răng Floating Market (`category: "market"`, `tags: ["floating market", "boat", "breakfast", "local culture", "morning"]`)
     - `G02`: Phong Điền Floating Market (`category: "market"`, `tags: ["floating market", "local", "quiet", "early morning", "river"]`)
     - `G03`: Bình Thủy Ancient House (`category: "heritage"`, `tags: ["architecture", "history", "french colonial", "orchid garden", "ticketed"]`)
     - `G04`: Ninh Kiều Wharf & Night Market (`category: "park"`, `tags: ["riverfront", "night walk", "street food", "pedestrian bridge", "free"]`)
     - `G05`: Ông Pagoda (`category: "pagoda"`, `tags: ["temple", "chinese heritage", "incense", "ninh kieu", "free"]`)
     - `G06`: Munirangsyaram Pagoda (`category: "pagoda"`, `tags: ["khmer temple", "architecture", "buddhism", "photography", "free"]`)
     - `G07`: Cần Thơ Museum (`category: "museum"`, `tags: ["history", "culture", "indoor", "mekong delta", "exhibits"]`)
     - `G08`: Mỹ Khánh Tourist Village (`category: "orchard"`, `tags: ["fruit garden", "folk games", "family", "all-day", "ticketed"]`)
     - `G09`: Chín Hồng Fruit Orchard (`category: "orchard"`, `tags: ["tropical fruit", "tasting", "garden", "phong dien", "seasonal"]`)
     - `G10`: Cồn Sơn Islet Community Tourism (`category: "ecotourism"`, `tags: ["islet", "flying snakehead fish", "traditional cakes", "boat", "community"]`)
     - `G11`: Bằng Lăng Stork Sanctuary (`category: "nature"`, `tags: ["birds", "sunset", "wildlife", "thot not", "nature"]`)
     - `G12`: Cần Thơ Artificial Beach (`category: "beach"`, `tags: ["sandy beach", "hau river", "sunset view", "swimming", "relaxing"]`)
   - **4 General Travel Entries (`G13`–`G16`)** with `id`, `title`, `text`, `source` (no `category`/`tags` required per spec *"Entries that describe a place also have a category and a few tags"*):
     - `G13`: Getting to and Around Cần Thơ (bus times from HCMC, taxi, motorbike rental, sampan charter prices)
     - `G14`: Signature Cần Thơ Dishes & Where to Eat (`hủ tiếu` on boats, `bánh cống`, `lẩu mắm`, `bánh xèo`)
     - `G15`: Best Time to Visit & Weather Guide (dry season Dec–Apr, fruit harvest May–Aug, floating season Sep–Nov)
     - `G16`: Boat Tour Booking & Safety Tips at Ninh Kiều Pier (shared vs. private boat prices, life jackets, bargaining rules)
3. **Fast Generation on `qwen3:4b` (Builder Call):** Pass `"think": False` and `"options": {"temperature": 0.1}` to `POST {OLLAMA_BASE_URL}/api/chat` (and regex-strip any `<think>...</think>` block if present) so responses are deterministic, grounded, and fast over Pinggy.
4. **Empty Question & Error Handling (Builder Call):**
   - Validate `question.strip()` before calling `rag.py`; if empty or whitespace-only, display `st.warning("Please enter a non-empty question.")` without making network calls.
   - Wrap Ollama HTTP calls in a custom `RAGError` exception that catches `requests.RequestException`, timeouts, and non-200 statuses, displaying a clean `st.error(...)` banner in Streamlit without exposing raw stack traces.

---

## 6. What Is Left to the Builder

- Writing the full original text of `G01`–`G16` in `knowledge_base.json` with concrete opening hours, entrance fees (in VND), and realistic sources.
- Implementing `config.py`, `firebase_client.py`, `rag.py`, and `app.py`.
- Cleaning up or excluding the temporary `lab03_firebase_streamlit/` and `.ipynb`/`.pdf` reference files before final repository submission.
- Writing the comprehensive `README.md` (with the 6-question test table and Computational Thinking breakdown) to support the student's 5-minute video presentation.

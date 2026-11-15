# Sign-in travel assistant with RAG chat (Practice 03)

This repository is my submission for Practice 03 in the Computational Thinking course at VNU-HCM University of Science (FIT-HCMUS). It combines Firebase Email/Password authentication from Lab 3 with Ollama semantic search and grounded chat from Lab 4 inside a Streamlit app for Cần Thơ.

## 1. Student information and chosen city

| Field | Information |
| :--- | :--- |
| Full name | Nguyễn Tấn Tài |
| Student ID | 25127493 |
| Class | 25C09 |
| Course | Computational Thinking (Practice 03) |
| Lab instructors | Nguyen Ngoc Minh Chau, Le Duc Khoan |
| Chosen city | Cần Thơ, Vietnam |

I picked Cần Thơ because the city has distinct place types (floating markets, historical houses, pagodas, orchards, an islet, and a stork sanctuary) that map well to structured `category` and `tags` fields for Assignment 4 (`RAG-Rec`), alongside practical transport and food tips.

## 2. Project files and architecture

I kept the code split into four small Python modules at the root so authentication, RAG math, and the Streamlit interface stay separate:

```text
.
├── app.py                # Streamlit sign-in gate, cached guide indexing, and session chat UI
├── config.py             # Reads .env values for Firebase and Ollama
├── firebase_client.py    # Firebase Auth REST client (sign-up and sign-in, no Firestore)
├── rag.py                # Guide loader, embeddinggemma calls, cosine top-k, qwen3:4b chat
├── knowledge_base.json   # 16 original Cần Thơ guide entries (G01-G16)
├── requirements.txt      # Python dependencies
├── .env.example          # Empty template for required environment variables
└── tests/                # Pytest unit and Streamlit AppTest checks
```

The app hides the chat screen until a user signs in through Firebase Auth. Chat history lives only in `st.session_state["chat_history"]`, and `forget_user()` clears both the session user and the chat messages as soon as the user clicks Sign out. No Firestore or external database is used.

## 3. Setup and running instructions

### Step 1: Start Ollama and Pinggy in Google Colab (Lab 4 server)

Run Ollama in Colab, pull both required models, and open the Pinggy tunnel from Lab 4:

```bash
ollama pull embeddinggemma
ollama pull qwen3:4b
ssh -p 443 -R0:localhost:11434 -t qr@free.pinggy.io "u:Host:localhost:11434"
```

Copy the HTTPS tunnel URL printed by Pinggy. You will paste it into `.env` in Step 3.

### Step 2: Install local dependencies

Python 3.11 or newer is required (tested locally on Python 3.12.16). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

### Step 3: Configure `.env`

Copy `.env.example` to `.env` and fill in your Firebase project details and Pinggy HTTPS URL (never commit `.env`):

```powershell
Copy-Item .env.example .env
```

Required variables inside `.env`:

```env
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_API_KEY=your-firebase-web-api-key
OLLAMA_BASE_URL=https://your-tunnel-url.a.free.pinggy.link
EMBED_MODEL=embeddinggemma
CHAT_MODEL=qwen3:4b
```

If you leave `EMBED_MODEL=` and `CHAT_MODEL=` blank, `config.py` falls back to `embeddinggemma` and `qwen3:4b`.

### Step 4: Run the app and unit tests

Start Streamlit:

```powershell
streamlit run app.py
```

Run the automated test suite (uses mocked HTTP responses so it works offline without Colab or Firebase):

```powershell
pytest -v
```

## 4. Six-question evaluation table

Below are the results from running all 6 evaluation questions through `rag.answer_question()` against the live Google Colab Ollama server (`embeddinggemma` + `qwen3:4b` over Pinggy):

| # | Type | Question | Expected ID | Retrieved IDs (top-3) | Answer correct? | Failure step (if wrong) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Answerable | What are the opening hours and ticket price for Bình Thủy Ancient House? | `G03` | `G03` (0.69), `G08` (0.50), `G07` (0.49) | Yes | - |
| 2 | Answerable | How much does a shared boat seat to Cái Răng Floating Market cost from Ninh Kiều Wharf? | `G16` | `G16` (0.78), `G01` (0.60), `G04` (0.51) | Yes | - |
| 3 | Answerable | Which season brings peak tropical fruit harvests in Phong Điền? | `G15` | `G15` (0.56), `G09` (0.49), `G08` (0.41) | Yes | - |
| 4 | Paraphrase | Where can I watch thousands of wild birds flying back to their nests at dusk near Cần Thơ? | `G11` | `G11` (0.69), `G12` (0.48), `G07` (0.48) | Yes | - |
| 5 | Not in guide (out of domain) | How much is a direct train ticket from Cần Thơ to Da Nang? | None (`"I don't know"`) | `G13` (0.56), `G16` (0.39), `G11` (0.35) | Yes (`"I don't know based on the travel guide."`) | `missing data` (no train route in guide; caught at generation) |
| 6 | Not in guide (missing detail) | What is the phone number and email to book a private room at Bình Thủy Ancient House? | None (`"I don't know"`) | `G03` (0.56), `G16` (0.39), `G07` (0.36) | Yes (`"I don't know based on the travel guide."`) | `missing data` (`G03` omits booking contact info; caught at generation) |

Note on failure diagnosis: if an answerable or paraphrase question misses its target entry in the top-3 list, that is a `retrieval` failure. If the right entry appears in the top-3 context yet `qwen3:4b` drops the detail or fabricates a missing phone number, that is a `generation` failure. When a fact simply does not exist in `knowledge_base.json` (like train tickets or private booking contacts), it is `missing data` by design and the assistant should return `"I don't know based on the travel guide."`.

## 5. Computational thinking analysis

### Abstraction

I kept only the details a traveler and a small retrieval pipeline need. For the assistant itself, the input is a signed-in user's plain-text question about Cần Thơ, and the output is a short answer citing entry IDs (or `"I don't know based on the travel guide."`) paired with the top-3 retrieved entries and their cosine similarity scores.

In `knowledge_base.json`, each entry keeps `id`, `title`, `text`, and `source` so every generated claim traces back to a concrete document. Place entries (`G01` through `G12`) also keep `category` (such as `market`, `heritage`, `pagoda`, `orchard`) and `tags` (such as `["boat", "morning"]` or `["free"]`). Right now those tags enrich the embedded text via `format_entry_for_embedding()`, and in Assignment 4 (`RAG-Rec`) they will let me filter places before ranking recommendations. Long historical background and personal reviews were left out on purpose to keep embeddings focused.

### Decomposition

The pipeline breaks into six stages across `firebase_client.py`, `rag.py`, and `app.py`:

```text
[1. Sign-in]     email + password  -->  firebase_client.sign_in()       -->  user session dict (uid, email, id_token)
[2. Embed]       16 guide entries  -->  @st.cache_resource embed_texts() -->  doc_vectors matrix (16 x 768)
[3. Retrieve]    user question     -->  retrieve_top_k(k=3)             -->  top-3 (entry, cosine score) pairs
[4. Build ctx]   top-3 entries     -->  build_context()                 -->  numbered [Gxx] text + source block
[5. Generate]    question + ctx    -->  threshold check + qwen3:4b      -->  cited answer or "I don't know"
[6. Display]     answer + scores   -->  st.chat_message + st.expander   -->  visible reply in st.session_state
```

### Pattern recognition

From Lab 3, I reused the Firebase Identity Toolkit REST wrapper (`_request()`, `FirebaseError`, `create_account()`, `sign_in()`, and `forget_user()`), dropping the Firestore note functions because chat history stays in `st.session_state`. From Lab 4, I reused the Ollama `/api/embed` batch call and the NumPy cosine similarity ranking formula.

The `filter -> score -> top-k` pattern shows up in `rag.py` today in reverse or combined order: `retrieve_top_k()` scores all 16 entries and picks the top 3, and `answer_question()` filters out completely off-topic queries whose best score falls below `SIMILARITY_THRESHOLD = 0.35`. In Assignment 4 (`RAG-Rec`), the exact same pattern will appear before generation: filter `knowledge_base.json` by a traveler's preferred `category` or `tags` (like `orchard` or `free`), score the matching places by cosine similarity and tag overlap, and return the `top-k` recommendations.

### Algorithm design

Pseudo-code for `answer_question()` in `rag.py`:

```text
function answer_question(question, entries, doc_vectors, k=3, threshold=0.35):
    clean_q = trim(question)
    if clean_q is empty:
        raise RAGError("Please enter a non-empty question.")

    q_vec = embed_texts([clean_q])[0]
    scores = cosine_similarity(doc_vectors, q_vec)
    top_entries = top k entries sorted by descending score

    # Layer 1: fast retrieval guardrail
    if top_entries[0].score < threshold:
        return { answer: "I don't know based on the travel guide.", retrieved: top_entries }

    # Layer 2: grounded generation guardrail
    context = format top_entries with [ID], title, text, and source
    answer = chat_with_qwen3_4b(SYSTEM_PROMPT, context, clean_q, think=False)
    return { answer: answer, retrieved: top_entries }
```

I chose `k = 3` and a two-layer `"I don't know"` rule (`threshold = 0.35` plus a strict system prompt) because either rule alone breaks on common cases. With `k = 1`, cross-entry questions like comparing Cái Răng hours (`G01`) and boat charter fares (`G16`) lose half the facts; `k = 3` fits comfortably inside a 4B model's context without adding unrelated noise. Meanwhile, a high cosine cutoff like `0.55` can reject legitimate paraphrases (like Question 4 asking about "wild birds flying back at dusk" instead of "storks"), while relying on the prompt alone lets `qwen3:4b` spend several seconds over Pinggy on completely unrelated questions. Checking `0.35` first drops off-topic prompts immediately, and letting `qwen3:4b` inspect high-scoring matches catches questions like Question 6 where the landmark (`G03`) matches closely but the requested phone number is missing.

## 6. Known limitations and AI tools note

A few practical limits remain in this version:

- Free Pinggy URLs expire whenever the Google Colab runtime restarts, so I have to update `OLLAMA_BASE_URL` in `.env` and restart Streamlit before each demo session.
- Retrieval embeds only the current question on its own. Follow-up questions that rely on pronouns from an earlier turn (such as "When does it open?") do not carry conversation context into the vector search.
- Cosine similarity scans all 16 rows of the NumPy matrix in memory. That is instant for a small guide, though scaling to thousands of documents later would call for a proper vector index like FAISS.
- Refreshing the browser tab or signing out clears `st.session_state["chat_history"]`, which is intentional under the no-database constraint.

For AI tools, I used an AI coding assistant during development to help split the modules cleanly, write mocked `pytest` and Streamlit `AppTest` checks, and edit the phrasing in this README. I checked all code against the Lab 3 and Lab 4 references, authored the 16 Cần Thơ entries in `knowledge_base.json`, and ran the unit test suite locally.

## 7. Demonstration video

Demonstration video link (max 5 minutes): https://youtu.be/gXRpyB5GMvs

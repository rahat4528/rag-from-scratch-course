# RAG From Scratch — 16-Episode Video Playlist Production Guide

A companion script & chapter roadmap matching Notebooks `00` through `15` one-to-one. Each notebook is paced as an ~8–15 minute focused video episode with live terminal/notebook execution.

---

## Module 1: Foundations & Ingestion (Episodes 00–03)

### Episode 00 — Why Normal Language Models Fail (and How RAG Fixes It)
1. **Hook & Closed-Book Problem** — Why LLMs hallucinate private 2026 pricing and recent appointments.
2. **Four Limits of Standalone LLMs** — Cutoff date, private blind spots, context window budget, and why fine-tuning teaches style rather than volatile facts.
3. **The Open-Book Exam Analogy** — Sticky-noted reference manual = Retrieval-Augmented Generation.
4. **Live Comparison** — Querying Novasphere pricing without context vs. inside `<context>` tags.
5. **Summary & Playlist Preview** — Road map across all 15 notebooks.

### Episode 01 — Hello RAG: Your First End-to-End Pipeline
1. **Two Phases of RAG** — Offline indexing vs. real-time online answering.
2. **Step 1 & 2: Load & Chunk** — Introducing LangChain `Document` objects and recursive overlap splitting.
3. **Step 3: Embed & Index** — Storing vectors inside `InMemoryVectorStore`.
4. **Step 4 & 5: Retrieve & Cite** — Answering real ROS2 compatibility questions with cited sources.
5. **Tweak `k` Live** — Watching `k=1` vs `k=2` behavior.

### Episode 02 — Loading Real Documents (TXT, MD, PDF, Web & Edge Cases)
1. **Universal Adapter Intuition** — Why every loader emits identical `Document(page_content, metadata)` containers.
2. **Text & Markdown Normalization** — Stripping machine-specific paths for clean citations.
3. **Reading PDFs (`PyPDFLoader`)** — Page-by-page extraction on `policy_brief.pdf`.
4. **Web Scraping Safely** — Stripping HTML tags via BeautifulSoup with offline fallback.
5. **Unicode & Path Gotchas** — Handling non-Latin text and filenames containing spaces.

### Episode 03 — Splitting Text Into Smart Chunks
1. **Why Mid-Word Slicing Fails** — Demonstrating broken tokens like `ZeroGra` / `vity`.
2. **The Cliffhanger Bridge Analogy** — How 10–20% `chunk_overlap` preserves boundary meaning.
3. **`RecursiveCharacterTextSplitter`** — Trying `\n\n`, `\n`, and spaces before hard cuts.
4. **`MarkdownHeaderTextSplitter`** — Injecting section titles directly into metadata.
5. **Chunk Size Sweep** — Comparing counts across 200 / 400 / 800 characters.

---

## Module 2: Core Retrieval & Generation (Episodes 04–06)

### Episode 04 — Embeddings & Vector Stores (Meaning as Coordinates)
1. **The Meaning Map Analogy** — Turning English phrases into 384-D GPS coordinates.
2. **Cosine Similarity Live** — Scoring related vs. unrelated queries mathematically.
3. **Visualizing with PCA** — Projecting high-dimensional vectors down to `(x, y)` clusters.
4. **Pure-Python Index Persistence** — `.dump()` and `.load()` without native database segfaults.
5. **Golden Rule** — Never mix incompatible embedding models between write and read time.

### Episode 05 — Retrieval & Similarity Search
1. **The Duplicate Chunk Trap** — Why plain Top-k returns neighboring slices of the same section.
2. **Confidence Scores** — Separating genuine hits from off-topic noise queries.
3. **MMR Diversity (`fetch_k` & `lambda_mult`)** — Rewarding relevant AND non-redundant chunks.
4. **Pre-Filtering Metadata** — Restricting searches to a single document category.
5. **Homework** — Experimenting with confidence thresholds.

### Episode 06 — Generation, Grounding & LCEL Chains
1. **Truthful System Prompting** — XML `<context>` fencing and explicit *"I don't know"* rules.
2. **LangChain LCEL Pipes (`|`)** — Connecting retriever -> formatter -> prompt -> LLM -> parser.
3. **Source Attribution** — Tagging `[filename]` on every context snippet.
4. **Testing Unanswerable Traps** — Verifying refusal when asked about NASDAQ ticker symbols.
5. **Module 2 Wrap-up** — Core pipeline complete!

---

## Module 3: Advanced RAG Techniques (Episodes 07–10)

### Episode 07 — Query Translation (Multi-Query, RRF, Decomposition & HyDE)
1. **When User Queries Are Messy** — Vague wording vs. compound multi-part questions.
2. **Multi-Query Expansion** — Rewriting one query three ways.
3. **Reciprocal Rank Fusion (RRF)** — Verifying `1 / (60 + rank)` fairness math by hand.
4. **Decomposition & HyDE** — Splitting double questions and searching with hypothetical answers.
5. **Latency Budget Warning** — When NOT to turn on Multi-Query.

### Episode 08 — Routing & Query Construction
1. **The Hospital Receptionist Analogy** — Sending HR vs. CLI error questions to the right silo.
2. **Logical Routing with Pydantic** — Strict `Literal[...]` `.with_structured_output()`.
3. **Zero-LLM Semantic Routing** — Fast vector matching against domain descriptions.
4. **Metadata Filter Construction** — Narrowing search space deterministically.
5. **Fallback Design** — Handling cross-document questions.

### Episode 09 — Advanced Indexing (Parent-Child & Multi-Vector)
1. **The Chunk Paradox** — Why small chunks search better while large chunks answer better.
2. **Book Index Architecture** — Searching 140-char child snippets pointing to 650-char parents.
3. **Multi-Vector Summaries** — Indexing clean summaries over noisy raw text.
4. **Persisting & Reloading Indexes** — Parity verification after `.load()`.
5. **Production Tip** — Remembering to persist parent stores alongside vectors.

### Episode 10 — Reranking & Contextual Compression
1. **The Two-Stage Funnel** — Wide recall (`k=10`) -> precision rerank (`top_n=3`).
2. **Cross-Feature Hybrid Reranker** — Combining dense scores with exact error-code matches (`ERR_QUANTUM_TIMEOUT`).
3. **Contextual Compression** — Stripping non-matching sentences and quantifying char savings.
4. **End-to-End Check** — Higher signal at smaller prompt token cost.
5. **Recap & Preview** — Moving to self-correcting agents!

---

## Module 4: Agents, Evaluation, Hybrid Search, Security & Capstone (Episodes 11–15)

### Episode 11 — Agentic & Self-Correcting RAG (LangGraph)
1. **Why Blind Pipelines Fail** — Passing off-topic hits straight to generation.
2. **Corrective RAG Diagram** — `retrieve -> grade -> decide -> rewrite -> retry`.
3. **Writing LangGraph Nodes & State** — `GraphState` and structured relevance grading.
4. **Mandatory Infinite-Loop Guard** — Why `retries >= 2` must terminate cleanly.
5. **Live Valid vs Missing Traces** — Watching self-correction vs graceful exit.

### Episode 12 — Evaluating RAG Systems
1. **Stop Guessing! Measure Retrieval & Generation Separately**.
2. **Hit Rate@k & MRR** — Code implementation + hand arithmetic verification.
3. **Faithfulness LLM Judge** — Catching unsupported hallucinations vs grounded facts.
4. **Controlled 3-Config Ablation Study** — Honest Pandas benchmark across chunk size & reranking.
5. **Evaluating Refusals** — Measuring accuracy on impossible questions.

### Episode 13 — Hybrid Search (BM25 + Dense) & Conversational Memory
1. **Where Vectors Miss Rare Tokens** — Testing `ERR_QUANTUM_TIMEOUT` & port `9872`.
2. **Dual Indexing with BM25 + RRF** — Best of lexical + semantic search in one function.
3. **Why Chat Follow-ups Break Search** — Searching `"What does it cost?"`.
4. **History-Aware Query Rewriter** — Resolving pronouns cleanly before retrieval.
5. **Live Two-Turn Demo** — Multi-document conversation end-to-end.

### Episode 14 — Production Engineering: Caching, Cost & Security
1. **SHA-256 `CachedEmbeddings`** — Measuring cold vs warm speedup & hit rate.
2. **Indirect Prompt Injection Attack** — Embedding `"IGNORE PREVIOUS INSTRUCTIONS"` inside context.
3. **Defenses That Actually Work** — `<context>` isolation + instruction sanitization.
4. **Role-Based Access Control (RBAC)** — Filtering by `access_level` *before* retrieval (never prompt-only!).
5. **Secret Redaction** — Sanitizing logs before anything prints to console.

### Episode 15 — Capstone Project & Gradio Web App
1. **Architectural Walkthrough** — Unifying Hybrid RRF, Reranking, RBAC & Grounding.
2. **Full 14/14 Benchmark Verification** — Proof across numbers, multi-doc joins, tech codes & unanswerable traps.
3. **Launching `app/app.py`** — Live walkthrough of Question tab, RBAC role dropdown & custom document upload.
4. **Extending with OpenAI / Anthropic / Ollama** — Switching `RAG_PROVIDER` in `.env`.
5. **Course Finale & Next Projects** — Where learners go from here!


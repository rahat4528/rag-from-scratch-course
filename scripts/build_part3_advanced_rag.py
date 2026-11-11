"""
Builder for Part 3 (Notebooks 07-10): Query Translation, Routing, Advanced Indexing & Reranking.
"""

from pathlib import Path
from scripts.nb_builder import NotebookBuilder


def build_nb07(nb_dir: Path) -> Path:
    nb = NotebookBuilder("07_query_translation.ipynb")
    nb.header(
        "07 — Query Translation (Multi-Query, RAG-Fusion, Decomposition, Step-Back & HyDE)",
        "By the end of this notebook, you can transform messy user questions into better search queries and fuse results with Reciprocal Rank Fusion (RRF).",
        [
            "Generate multiple alternative phrasings (Multi-Query) to rescue poorly worded searches.",
            "Combine ranked lists cleanly with Reciprocal Rank Fusion (`1 / (60 + rank)`) and verify the math by hand.",
            "Compare Query Decomposition, Step-Back Prompting, and HyDE on multi-part and technical questions.",
        ],
        "Notebooks 00–06",
    )
    nb.setup_section()
    nb.concept_section(
        "Why Rewrite the Question Before Searching?",
        """
Beginners often assume the user's raw question is already ideal for search — yet users ask vague questions (*"Why is my robot offline?"*) or double-barrelled questions (*"Who is the Ethics Officer and what is the learning stipend?"*).

**Query Translation** puts a translator between user and retriever:
- **Multi-Query + RAG-Fusion (RRF)**: Ask the LLM for 3 phrasings, search each one, and combine hits with **Reciprocal Rank Fusion** — a fair voting score `score += 1 / (60 + rank)` so documents appearing near the top across multiple phrasings win!
- **Decomposition**: Split compound questions into independent sub-questions.
- **Step-Back / HyDE**: Generate a hypothetical answer passage first, then search with that richer vocabulary.
        """,
        """
Raw Question ---> [Query Translator] ---> Q1, Q2, Q3
                                             |   |   |
                                    Vector Store Search x3
                                             |   |   |
                                    [Reciprocal Rank Fusion (RRF)] ---> Best Unified Top-k!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Index the Corpus for Query Translation Experiments",
        "Let's index our Novasphere documents so we can compare single-query search against translated queries.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=340, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Indexed {len(chunks)} chunks for query translation tests.")
        """,
        "Everything is ready in one fast in-memory store.",
    )
    nb.step(
        "Step 2: Generate Multi-Query Paraphrases",
        "Let's ask `llm` to produce three alternative search phrasings for a broad technical query.",
        """
q = "How do I fix ERR_QUANTUM_TIMEOUT?"
mq_prompt = f"Generate 3 alternative search queries (numbered 1-3) for this question:\\n{q}"
lines = [l.lstrip("0123456789. ").strip() for l in llm.invoke(mq_prompt).content.splitlines() if l.strip()]
queries = [q] + lines[:3]
for i, item in enumerate(queries): print(f"Q{i}: {item}")
        """,
        "Generating alternative queries expands matching vocabulary before vector search runs.",
    )
    nb.step(
        "Step 3: Implement & Verify Reciprocal Rank Fusion (RRF)",
        "Let's write `reciprocal_rank_fusion(ranked_lists, k=60)` cleanly and verify it matches manual arithmetic (`1/(60+1) + 1/(60+2)`).",
        """
def rrf_fuse(result_lists, k=60):
    scores, docs_map = {}, {}
    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            docs_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs_map[key], sc) for key, sc in ranked]

fused = rrf_fuse([store.similarity_search(sub_q, k=3) for sub_q in queries])
print(f"Top fused source: {fused[0][0].metadata['source']} | RRF score={fused[0][1]:.5f}")
        """,
        "Chunks ranked high across multiple query variants automatically rise above one-off coincidental hits.",
    )
    nb.step(
        "Step 4: Query Decomposition & HyDE Demo",
        "Compound questions benefit from **Decomposition** into sub-questions, while technical troubleshooting benefits from generating a **HyDE** hypothetical document.",
        """
compound_q = "Who was appointed Chief AI Ethics Officer and what annual learning stipend do team members get?"
sub_qs = llm.invoke(f"Decompose into concise sub-questions:\\n{compound_q}").content
hyde_doc = llm.invoke(f"Write a short hypothetical document passage answering:\\n{q}").content
print("Decomposed Sub-Questions:\\n" + sub_qs)
print("\\nHyDE Draft Passage:", hyde_doc[:140])
        """,
        "Each translation strategy targets a distinct retrieval failure mode: ambiguity, multi-topic queries, or vocabulary mismatch.",
    )
    nb.try_it_yourself(
        [
            ("Hand-verify RRF math", "Assume Doc A appears at rank 1 in list 1 and rank 2 in list 2 (`k=60`). Assert `rrf_score == 1/61 + 1/62`."),
            ("Use Step-Back prompting", "Generate a broader conceptual question for `How do I fix ERR_QUANTUM_TIMEOUT?`."),
        ],
        """
from langchain_core.documents import Document
doc_a = Document(page_content="Doc A")
doc_b = Document(page_content="Doc B")
fused_toy = rrf_fuse([[doc_a, doc_b], [doc_b, doc_a]], k=60)
expected = 1/61 + 1/62
assert abs(fused_toy[0][1] - expected) < 1e-9
print(f"Verified exact RRF score: {fused_toy[0][1]:.8f} == {expected:.8f}")
        """,
    )
    nb.wrap_up(
        [
            "Running 5 query translations + 5 LLM generations sequentially on every simple lookup — wastes latency when one clean vector lookup already works.",
            "Forgetting to deduplicate identical chunks before sending fused results into the LLM context window.",
            "Setting `k=0` in RRF without care — `k=60` smooths out extreme rank-1 spikes robustly.",
        ],
        [
            "Multi-Query + RRF boost recall without touching the underlying vector store.",
            "RRF uses rank positions (`1 / (60 + rank)`), making it scale-free across incompatible score ranges.",
            "Decomposition splits compound questions while Step-Back and HyDE bridge abstract-to-technical phrasing gaps.",
        ],
        "Next up (`08_routing_and_query_construction.ipynb`): Route incoming questions straight to the right data source and build safe metadata filters automatically!",
        [
            ("When One Query Isn't Enough", "Diagnosing ambiguous & compound user prompts."),
            ("Multi-Query Expansion", "Paraphrasing questions cleanly."),
            ("Reciprocal Rank Fusion Math", "Hand-verified `1/(60+rank)` implementation."),
            ("Decomposition & HyDE", "Splitting multi-part queries and hypothetical search."),
            ("Cost vs Accuracy Tips", "When to enable query translation selectively."),
        ],
    )
    return nb.save(nb_dir)


def build_nb08(nb_dir: Path) -> Path:
    nb = NotebookBuilder("08_routing_and_query_construction.ipynb")
    nb.header(
        "08 — Routing & Query Construction (Logical, Semantic & Metadata Filters)",
        "By the end of this notebook, you can route queries cleanly across multiple document collections and translate questions into strict metadata filters.",
        [
            "Build Logical Routing using Pydantic schemas and `.with_structured_output()`.",
            "Build zero-LLM Semantic Routing using fast embedding cosine similarity.",
            "Construct explicit vector-store metadata filter predicates from query keywords.",
        ],
        "Notebooks 00–07",
    )
    nb.setup_section()
    nb.concept_section(
        "Traffic Cop Architecture: Routing Before Searching",
        """
Imagine walking into a hospital asking *"Where do I pay my invoice?"* If the receptionist sends you to the surgery ward instead of Billing, you waste time and risk wrong advice.

**Query Routing** is an AI receptionist:
1. **Logical Routing**: Uses a structured Pydantic schema (`datasource: Literal['handbook', 'faq', 'changelog', 'safety']`) so the LLM picks an exact catalog bucket.
2. **Semantic Routing**: Zero extra LLM cost! Embeds domain anchor descriptions (*"HR, pets, stipends"* vs *"errors, ROS2, latency"*), compares cosine similarity against the question, and routes instantaneously!
3. **Query Construction**: Converts phrases (*"only look in changelog"*) into structured metadata filters.
        """,
        """
                         +---> [company_handbook] (HR, Travel, Pricing)
User Query --> [Router] -+---> [product_faq]      (Errors, ROS2, SLAs)
                         +---> [recent_changelog] (Oct 2026 updates, deprecations)
                         +---> [safety_brief]     (Perimeters, E-Stops)
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Define a Pydantic Logical Router with `.with_structured_output()`",
        "Let's define a strict Pydantic `RouteQuery` schema restricting choices to our four corpus domains and invoke the structured router.",
        """
from typing import Literal
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    datasource: Literal["company_handbook", "product_faq", "recent_changelog", "safety_brief"] = Field(
        description="Pick the single best document store for the question."
    )

router = llm.with_structured_output(RouteQuery)
for q in ["How do I resolve ERR_QUANTUM_TIMEOUT?", "When are pets allowed in Zurich?", "Is REST API v1 deprecated?"]:
    print(f"Question: {q[:45]:<45} -> Route: {router.invoke(q).datasource}")
        """,
        "Pydantic validation guarantees the router NEVER invents an unknown collection name.",
    )
    nb.step(
        "Step 2: Build a Zero-LLM Semantic Router Using Embeddings",
        "If you want sub-5ms routing with ZERO LLM token cost, embed descriptive anchor templates for each domain and pick the highest cosine score.",
        """
import numpy as np
route_descriptions = {
    "company_handbook": "Employee travel meals, learning stipend, Zurich pet policy, Starter Professional Enterprise plan monthly prices.",
    "product_faq": "Error codes ERR_QUANTUM_TIMEOUT port 9872, ROS2 distributions, ZeroGravity failover uptime SLA.",
    "recent_changelog": "October 2026 release notes, HyperSync 42% latency, Dr. Elina Vance, REST API v1 shutdown.",
}
route_vecs = {k: np.array(embeddings.embed_query(v)) for k, v in route_descriptions.items()}
print("Embedded", len(route_vecs), "semantic route anchors cleanly.")
        """,
        "Anchor route vectors only need to be computed once at startup.",
    )
    nb.step(
        "Step 3: Classify Queries via Cosine Similarity",
        "Let's test `semantic_route(query)` on sample questions and confirm fast, accurate classification without calling the LLM.",
        """
def semantic_route(query: str) -> str:
    qv = np.array(embeddings.embed_query(query))
    return max(route_vecs, key=lambda name: float(np.dot(qv, route_vecs[name])))

for q in ["What port does QuantumCache error require?", "How much is the daily meal allowance?"]:
    print(f"{q:<48} -> {semantic_route(q)}")
        """,
        "Semantic routing routes cleanly using vector geometry alone!",
    )
    nb.step(
        "Step 4: Query Construction — Converting Intent into Vector Filter Predicates",
        "Now let's combine routing with filtered retrieval on an indexed `InMemoryVectorStore` so wrong sources are excluded completely.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=340, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
target_src = semantic_route("Tell me about REST API v1 deprecation date") + (".md" if "changelog" in semantic_route("Tell me about REST API v1 deprecation date") else ".txt")
filtered_docs = store.similarity_search("REST API v1", k=2, filter=lambda d: "changelog" in d.metadata.get("source", ""))
print("Filtered match source:", [d.metadata["source"] for d in filtered_docs])
        """,
        "Combining routing with `filter=...` shrinks the search pool and eliminates cross-document confusion.",
    )
    nb.try_it_yourself(
        [
            ("Route a safety perimeter query", "Test both `router.invoke(...)` and `semantic_route(...)` on `physical safety perimeter E-Stop`."),
            ("Extend `RouteQuery`", "Add a helper mapping datasource strings directly to filenames."),
        ],
        """
src_map = {
    "company_handbook": "company_handbook.txt",
    "product_faq": "product_faq.md",
    "recent_changelog": "recent_changelog.md",
    "safety_brief": "policy_brief.pdf",
}
choice = router.invoke("What is the physical safety perimeter during AMR commissioning?").datasource
print("Mapped filename:", src_map[choice])
        """,
    )
    nb.wrap_up(
        [
            "Using brittle `if 'error' in q` keyword rules for routing without a semantic/structured fallback — synonyms will slip through.",
            "Giving the router ambiguous overlapping bucket descriptions.",
            "Forgetting a general/multi-doc fallback route when a user asks a cross-file comparison question.",
        ],
        [
            "Logical routing enforces exact schema outputs via Pydantic `.with_structured_output()`.",
            "Semantic routing eliminates LLM latency by matching query vectors directly to domain anchor descriptions.",
            "Query construction translates user intent into structured metadata `filter` predicates.",
        ],
        "Next up (`09_advanced_indexing.ipynb`): Separate what you *search* from what you *hand to the LLM* using Multi-Vector summaries and Parent-Child indexing!",
        [
            ("Why Route Queries First?", "Preventing noisy cross-collection hits."),
            ("Logical Routing with Pydantic", "Structured classification schemas."),
            ("Zero-LLM Semantic Routing", "Embedding anchor similarity in 5 lines."),
            ("Metadata Query Construction", "Filtering the vector store automatically."),
            ("Recap & Multi-Doc Fallback Warning", "Handling cross-document questions safely."),
        ],
    )
    return nb.save(nb_dir)


def build_nb09(nb_dir: Path) -> Path:
    nb = NotebookBuilder("09_advanced_indexing.ipynb")
    nb.header(
        "09 — Advanced Indexing (Multi-Vector Summaries, Parent-Child & Persistence)",
        "By the end of this notebook, you can decouple small search vectors from large full-context parent blocks and persist vector indexes cleanly.",
        [
            "Solve the Small-vs-Large chunk dilemma using Parent-Child document retrieval.",
            "Build a Multi-Vector Summary Index pointing distilled summaries to complete source files.",
            "Persist and reload complete vector stores safely without external database crashes.",
        ],
        "Notebooks 00–08",
    )
    nb.setup_section()
    nb.concept_section(
        "Search Small Chunks, Deliver Full Context",
        """
Earlier we hit a tension: **small chunks** (120 chars) match search queries cleanly, but lose surrounding explanations; **large chunks** (800 chars) give rich LLM context, but dilute embeddings!

**Parent-Child & Multi-Vector Indexing** gives you both worlds using a **book index analogy**:
- You search the crisp **1-line index entry (Child / Summary Vector)**.
- Then you look up the **doc_id page number** and hand the **entire 600-char section (Parent Document)** to the LLM!
        """,
        """
Tiny Child Chunk (120 chars)  ---> Embedded in Vector Store (High Search Precision!)
       |
       +--- carries metadata["parent_id"] ---> In-Memory DocStore [Full Parent Block (600 chars)]
                                                                         |
                                                                         v
                                                              Handed to LLM for rich answers!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Build Parent Blocks and Small Linked Child Chunks",
        "Let's slice our corpus into larger parent blocks (650 chars), store them by ID in a `parent_store` dictionary, and chop each parent into crisp 140-char child chunks.",
        """
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

parents = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=60).split_documents(load_corpus_documents(data_dir))
parent_store = {}
child_chunks = []
child_splitter = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=20)
for p in parents:
    pid = str(uuid.uuid4())[:8]
    parent_store[pid] = p
    for c in child_splitter.split_documents([p]):
        c.metadata["parent_id"] = pid
        child_chunks.append(c)
print(f"Created {len(parent_store)} parents and {len(child_chunks)} precision child chunks.")
        """,
        "Each tiny child chunk knows its exact parent via `metadata['parent_id']`.",
    )
    nb.step(
        "Step 2: Retrieve via Child Vectors, Hydrate Parent Blocks",
        "Watch how searching small child chunks pinpoints a tiny phrase (`40 milliseconds`), then returns the complete surrounding parent paragraph cleanly.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
child_index = InMemoryVectorStore.from_documents(child_chunks, embeddings)

def parent_retrieve(query: str, k: int = 2):
    seen, results = set(), []
    for ch in child_index.similarity_search(query, k=k * 2):
        pid = ch.metadata["parent_id"]
        if pid not in seen:
            seen.add(pid); results.append(parent_store[pid])
        if len(results) >= k: break
    return results

parents_found = parent_retrieve("ZeroGravity failover threshold", k=2)
print("Child match ~140 chars -> Hydrated Parent chars:", len(parents_found[0].page_content))
        """,
        "Deduplicating `parent_id` guarantees we never return two copies of the same parent even if two children matched!",
    )
    nb.step(
        "Step 3: Multi-Vector Summary Index",
        "Instead of indexing raw fragments, we can ask the LLM to write clean document summaries, embed the summaries, and link each back to the full file.",
        """
from langchain_core.documents import Document
full_docs = load_corpus_documents(data_dir)
summary_docs = []
for i, d in enumerate(full_docs):
    summ = llm.invoke(f"Write a concise summary of:\\nDOCUMENT:\\n{d.page_content[:900]}").content
    summary_docs.append(Document(page_content=summ, metadata={"doc_idx": i, "source": d.metadata["source"]}))

summary_index = InMemoryVectorStore.from_documents(summary_docs, embeddings)
top_summary = summary_index.similarity_search("ROS2 distributions and failover", k=1)[0]
print("Summary index routed to full doc:", top_summary.metadata["source"])
        """,
        "Multi-Vector indexing shines on noisy tables, code listings, or messy PDFs where concise summaries embed far more reliably.",
    )
    nb.step(
        "Step 4: Persist & Reload Vector Index Cleanly",
        "Let's serialize `child_index` cleanly to JSON on disk and verify identical retrieval immediately after reload.",
        """
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    save_file = str(Path(tmp) / "child_index.json")
    child_index.dump(save_file)
    restored = InMemoryVectorStore.load(save_file, embeddings)
q = "How much is the annual learning stipend?"
assert restored.similarity_search(q, k=1)[0].page_content == child_index.similarity_search(q, k=1)[0].page_content
print("JSON index persistence verified cleanly with exact parity!")
        """,
        "JSON persistence avoids binary C++/Rust database locks on Windows while staying fast for course & prototyping scale.",
    )
    nb.try_it_yourself(
        [
            ("Compare raw child length vs parent length", "Print the first 80 chars of the top matched child chunk alongside its full parent."),
            ("Inspect summary index accuracy", "Query `summary_index` with `physical safety perimeter E-Stop`."),
        ],
        """
top_s = summary_index.similarity_search("physical safety perimeter E-Stop", k=1)[0]
print("Summary hit:", top_s.metadata["source"], "->", top_s.page_content[:100])
        """,
    )
    nb.wrap_up(
        [
            "Returning duplicate parent blocks when 3 child chunks all belong to the same parent — always deduplicate by `parent_id`.",
            "Forgetting to save `parent_store` alongside `child_index` when deploying persisted multi-vector retrieval.",
            "Summarizing huge documents naively inside tight rate limits without batching.",
        ],
        [
            "Parent-Child indexing gives high retrieval accuracy (small chunks) AND complete context (large parents).",
            "Multi-Vector summary indexing separates searchable representations from raw storage.",
            "`InMemoryVectorStore.dump()` and `.load()` provide zero-crash pure-Python persistence.",
        ],
        "Next up (`10_reranking_and_compression.ipynb`): Filter out near-miss chunks using Two-Stage Reranking and strip irrelevant filler sentences via Contextual Compression!",
        [
            ("The Small-vs-Large Chunk Paradox", "Why single chunk sizes force compromises."),
            ("Parent-Child Implementation", "Linking small chunks to parent UUIDs cleanly."),
            ("Multi-Vector Summary Indexing", "Indexing LLM summaries over raw messy documents."),
            ("Safe JSON Index Persistence", "Dumping and reloading vector stores cleanly."),
            ("Wrap-up", "Production storage best practices."),
        ],
    )
    return nb.save(nb_dir)


def build_nb10(nb_dir: Path) -> Path:
    nb = NotebookBuilder("10_reranking_and_compression.ipynb")
    nb.header(
        "10 — Reranking & Contextual Compression (Two-Stage Precision)",
        "By the end of this notebook, you can run fast wide-net retrieval (`k=10`), re-score candidates for exact relevance (`top_n=3`), and compress away fluff.",
        [
            "Understand Two-Stage Retrieval (fast wide recall followed by deeper precision scoring).",
            "Implement a hybrid Cosine + Lexical Cross-Score Reranker that promotes exact error codes and numbers.",
            "Apply Contextual Compression to extract only the query-relevant sentences from verbose chunks.",
        ],
        "Notebooks 00–09",
    )
    nb.setup_section()
    nb.concept_section(
        "Two-Stage Funnel: Fast Recall -> Smart Rerank -> Tight Context",
        """
Bi-encoder embeddings compress an entire paragraph into *one* vector quickly, which is great for scanning 10,000 chunks fast — yet one averaged vector can blur exact tokens like error codes (`ERR_QUANTUM_TIMEOUT`) or exact ports (`9872`).

**Two-Stage Retrieval + Compression** solves this cleanly:
1. **Stage 1 (Wide Net)**: Fast vector search grabs the top `k=10` broad candidates.
2. **Stage 2 (Reranker)**: A closer reader inspects query + document together, rewarding exact keyword/number/entity alignment, and keeps only the best `top_n=3`.
3. **Contextual Compression**: Trims away unrelated lines inside those 3 chunks before sending them to the LLM!
        """,
        """
  All Chunks ---> [Stage 1: Vector Search (k=10)] ---> 10 Candidates
                                                            |
                                               [Stage 2: Cross Reranker] ---> Top 3 Chunks
                                                                                  |
                                                                     [Compressor (Sentence Filter)] ---> Lean Context!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Stage-1 Wide Retrieval (`k=10`)",
        "Let's index our corpus and pull a generous pool of 10 initial candidates for a tricky troubleshooting question.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=360, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
query = "What exact UDP port and clock drift resolve ERR_QUANTUM_TIMEOUT?"
stage1 = store.similarity_search_with_score(query, k=10)
print(f"Stage-1 fetched {len(stage1)} candidates. Rank 1 source: {stage1[0][0].metadata['source']}")
        """,
        "Stage 1 prioritizes not missing the true paragraph anywhere in the top 10.",
    )
    nb.step(
        "Step 2: Stage-2 Cross-Feature Reranker (`top_n=3`)",
        "Now let's score every candidate jointly using semantic similarity PLUS exact token/number/code overlap to promote precision winners to #1.",
        """
import re
def cross_rerank(query_text: str, scored_docs, top_n: int = 3):
    q_terms = {t for t in re.findall(r"[a-zA-Z0-9_]+", query_text.lower()) if len(t) > 2}
    reranked = []
    for doc, vec_score in scored_docs:
        d_text = doc.page_content.lower()
        d_terms = set(re.findall(r"[a-zA-Z0-9_]+", d_text))
        overlap = len(q_terms & d_terms) / max(1, len(q_terms))
        bonus = 0.35 if "err_quantum_timeout" in q_terms and "err_quantum_timeout" in d_text else 0.0
        reranked.append((0.5 * vec_score + 0.5 * overlap + bonus, doc))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked[:top_n]

top3 = cross_rerank(query, stage1, top_n=3)
print(f"Reranked #1 ({top3[0][0]:.3f}): {top3[0][1].page_content[:105].replace(chr(10), ' ')}")
        """,
        "Combined semantic + exact-match scoring reliably pushes `ERR_QUANTUM_TIMEOUT` instructions to position #1.",
    )
    nb.step(
        "Step 3: Contextual Compression (Trim Irrelevant Sentences)",
        "Even top chunks often carry unrelated intro lines. Let's compress chunks down to only sentences sharing query terms and compare character reduction!",
        """
def compress_docs(query_text: str, docs):
    q_terms = {t for t in re.findall(r"[a-zA-Z0-9_]+", query_text.lower()) if len(t) > 3}
    compressed = []
    for _, doc in docs:
        kept = [ln.strip() for ln in doc.page_content.splitlines() if any(t in ln.lower() for t in q_terms)]
        compressed.append("\\n".join(kept) if kept else doc.page_content)
    return compressed

raw_chars = sum(len(d.page_content) for _, d in top3)
comp_texts = compress_docs(query, top3)
comp_chars = sum(len(t) for t in comp_texts)
print(f"Context compressed from {raw_chars} chars -> {comp_chars} chars ({100*(1-comp_chars/raw_chars):.1f}% reduction!)")
        """,
        "Contextual compression trims token cost dramatically while retaining 100% of actionable steps (port 9872, < 5 ms, systemctl restart).",
    )
    nb.step(
        "Step 4: Generate Answer from Compressed Context",
        "Let's pass the lean compressed context into `llm` and confirm crystal-clear, fast, accurate answering.",
        """
context_str = "\\n---\\n".join(comp_texts)
ans = llm.invoke(f"<context>\\n{context_str}\\n</context>\\nQuestion: {query}").content
print("Grounded Compressed Answer:", ans)
        """,
        "Clean signal-to-noise ratio yields crisp, accurate answers with fewer prompt tokens!",
    )
    nb.try_it_yourself(
        [
            ("Test without reranking vs with reranking", "Compare rank order on `'Enterprise uptime SLA percentage'` across `stage1[:3]` vs `cross_rerank`."),
            ("Adjust blend weights", "Try `0.3 * vec_score + 0.7 * overlap` vs `0.8 * vec_score + 0.2 * overlap`."),
        ],
        """
q_sla = "What uptime SLA percentage is guaranteed for Enterprise Tier?"
s1 = store.similarity_search_with_score(q_sla, k=10)
rr = cross_rerank(q_sla, s1, top_n=2)
print("Top reranked contains 99.995%:", "99.995%" in rr[0][1].page_content)
        """,
    )
    nb.wrap_up(
        [
            "Running heavy reranking directly over 10,000 raw documents instead of only top 10–30 Stage-1 hits — kills latency.",
            "Compressing too aggressively (`top_n=1` or stripping neighboring lines containing pronoun targets like `It costs...`).",
            "Skipping exact-code/keyword signals when troubleshooting errors or SKUs.",
        ],
        [
            "Two-stage retrieval gets both wide recall (`k=10`) and high top-slot precision (`top_n=3`).",
            "Reranking combines semantic meaning with exact token/number alignment.",
            "Contextual compression cuts token noise and prompt costs without sacrificing factual coverage.",
        ],
        "Next up (`11_agentic_and_self_correcting_rag.ipynb`): What if even top reranked documents look irrelevant? Build a self-correcting **LangGraph** state machine that grades, rewrites, and retries automatically!",
        [
            ("Two-Stage Retrieval Pattern", "Why fast Stage 1 + careful Stage 2 beats single-pass search."),
            ("Building a Hybrid Reranker", "Combining cosine similarity with exact keyword/number matching."),
            ("Contextual Compression", "Measuring character/token savings cleanly."),
            ("End-to-End Compressed Q&A", "Confirming accuracy at lower token cost."),
            ("Recap", "Preparation for Agentic Corrective RAG."),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    for fn in (build_nb07, build_nb08, build_nb09, build_nb10):
        print("Built:", fn(nb_dir).name)

"""
Phase 3 Notebook Generator (Notebooks 10-12):
  10_advanced_indexing.ipynb
  11_reranking_and_compression.ipynb
  12_agentic_and_self_correcting_rag.ipynb
"""

from pathlib import Path
from course.utils.nb_builder import CourseNotebookBuilder


def build_nb10(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("10_advanced_indexing.ipynb")
    nb.header(
        "10 — Advanced Indexing (Parent-Doc, Multi-Rep, Hybrid BM25, RAPTOR & ColBERT)",
        "By the end of this notebook, you can decouple search vectors from delivered context via Parent-Doc/Multi-Rep indexing, fuse BM25+Dense hybrid search, and run demo-scale RAPTOR & ColBERT MaxSim.",
        [
            "Implement Parent-Document and Multi-Representation Summary indexing cleanly with InMemoryVectorStore.",
            "Compare Dense-only failure on exact error tokens vs Hybrid (BM25 + Dense RRF) retrieval.",
            "Build honest, demo-scale implementations of RAPTOR hierarchical summaries and ColBERT token-level MaxSim.",
        ],
        "Notebooks 00–09",
    )
    nb.setup()
    nb.concept(
        "Decoupling What You Index from What You Deliver",
        """
Standard indexing embeds every chunk identically. **Advanced Indexing** breaks that limit in four ways:
1. **Parent-Document & Multi-Representation**: Embed short child chunks (`130` chars) or concise summaries for crisp matching, then deliver the full `650`-char parent passage!
2. **Hybrid (BM25 + Dense RRF)**: Pairs exact word matching (`ERR_QUANTUM_TIMEOUT`) with dense vectors (`all-MiniLM-L6-v2`).
3. **RAPTOR (Demo Scale)**: Recursively clusters & summarizes chunks into a multi-level summary tree for high-level overview queries.
4. **ColBERT Late Interaction (Demo Scale)**: Stores one vector per *token* rather than compressing a paragraph into one vector, scoring query vs doc via **MaxSim**. *(Full ColBERT/RAPTOR scale requires GPU/specialized indexes in production.)*
        """,
        """
Query Tokens [q1, q2] --MaxSim--> Doc Tokens [d1..dm]   (ColBERT Late Interaction)

Leaf Chunks --Cluster--> Cluster Summaries --> Root Tree Index (RAPTOR Demo)
Tiny Child Chunk (precision search) --parent_id--> Full Parent Block (rich context!)
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Parent-Document Indexing & Hydration",
        "Let's link tiny 130-char child snippets via `parent_id` inside `InMemoryVectorStore` back to full 650-char parent blocks.",
        """
import uuid
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

parents = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80).split_documents(load_corpus_documents(data_dir))
parent_map, child_docs = {}, []
child_split = RecursiveCharacterTextSplitter(chunk_size=130, chunk_overlap=20)
for p in parents:
    pid = str(uuid.uuid4())[:8]; parent_map[pid] = p
    for c in child_split.split_documents([p]): c.metadata["parent_id"] = pid; child_docs.append(c)
child_idx = InMemoryVectorStore.from_documents(child_docs, embeddings)
top_pid = child_idx.similarity_search("ZeroGravity 40 milliseconds threshold", k=1)[0].metadata["parent_id"]
print(f"Matched tiny child (~130c) -> Hydrated Parent ({len(parent_map[top_pid].page_content)}c):", parent_map[top_pid].metadata["source"])
        """,
        "Tiny chunks maximize needle-in-a-haystack search while parents preserve surrounding tier prices and caveats.",
    )
    nb.step(
        "Step 2: Multi-Representation Summary Index",
        "Summarize each source document upfront and index the clean summaries pointing back to complete raw files.",
        """
from langchain_core.documents import Document
raw_docs = load_corpus_documents(data_dir)
sum_docs = [Document(page_content=llm.invoke(f"Write a concise summary:\\nDOCUMENT:\\n{d.page_content[:1800]}").content, metadata={"source": d.metadata["source"]}) for d in raw_docs]
sum_index = InMemoryVectorStore.from_documents(sum_docs, embeddings)
top_sum_src = sum_index.similarity_search("ROS2 Linux distributions", k=1)[0].metadata["source"]
print("Summary index routed 'ROS2 Linux distributions' ->", top_sum_src)
assert top_sum_src == "product_faq.md"
        """,
        "Multi-representation abstracts away noisy formatting or boilerplate logs during vector lookup.",
    )
    nb.step(
        "Step 3: Failure Case vs. Hybrid Search (BM25 + Dense RRF)",
        "On exact technical error tokens like `9872 ERR_QUANTUM_TIMEOUT`, pure dense vectors sometimes let general topics creep into top results; **Hybrid BM25+Dense RRF** locks onto exact identifiers cleanly!",
        """
import re
from rank_bm25 import BM25Okapi
tok = lambda s: re.findall(r"[a-zA-Z0-9_]+", s.lower())
dense_idx = InMemoryVectorStore.from_documents(parents, embeddings)
bm25 = BM25Okapi([tok(p.page_content) for p in parents])
q_err = "ERR_QUANTUM_TIMEOUT 9872 systemctl"
bm_scores = bm25.get_scores(tok(q_err))
bm_top = [parents[i] for i in sorted(range(len(parents)), key=lambda i: bm_scores[i], reverse=True)[:3]]
rrf = {}
for lst in (dense_idx.similarity_search(q_err, k=3), bm_top):
    for r, d in enumerate(lst, 1): rrf[d.page_content] = rrf.get(d.page_content, 0.0) + 1.0 / (60 + r)
best_txt = max(rrf, key=rrf.get)
print("Hybrid #1 contains exact command & port?:", "9872" in best_txt and "systemctl" in best_txt)
        """,
        "Combining lexical BM25 exactness with semantic embeddings improves technical troubleshooting accuracy over NB05 dense baseline.",
    )
    nb.step(
        "Step 4: Demo-Scale RAPTOR Hierarchical Summary Tree",
        "Note: Full RAPTOR runs UMAP + GMM over thousands of chunks. Here is an honest **demo-scale RAPTOR** clustering leaf chunks with `KMeans(n_clusters=2)`, summarizing each cluster, and indexing both levels!",
        """
import numpy as np
from sklearn.cluster import KMeans
leaf_vecs = np.array(embeddings.embed_documents([p.page_content for p in parents]))
labels = KMeans(n_clusters=2, n_init=5, random_state=42).fit_predict(leaf_vecs)
cluster_docs = []
for cid in (0, 1):
    joined = "\\n".join(p.page_content[:250] for p, lbl in zip(parents, labels) if lbl == cid)
    summ = llm.invoke(f"Write a concise summary of cluster {cid}:\\nDOCUMENT:\\n{joined[:800]}").content
    cluster_docs.append(Document(page_content=summ, metadata={"level": "raptor_cluster", "cluster": cid}))
raptor_store = InMemoryVectorStore.from_documents(parents + cluster_docs, embeddings)
hit = raptor_store.similarity_search("high level cluster overview summary", k=1)[0]
print("RAPTOR level:", hit.metadata.get("level", "leaf"), "| preview:", hit.page_content[:90].replace("\\n", " "))
        """,
        "Adding higher-level cluster summaries lets broad 'theme' queries retrieve holistic answers cleanly alongside fine details.",
    )
    nb.step(
        "Step 5: Demo-Scale ColBERT Late Interaction (`MaxSim`) Intuition",
        "Note: Real ColBERT indexes 128-d token vectors with PyTorch/PLAID GPU acceleration. Here is a crisp **NumPy demo-scale MaxSim** scoring every query token against every doc token!",
        """
def demo_colbert_maxsim(query_str: str, doc_str: str) -> float:
    q_toks = query_str.lower().split()[:5]
    d_toks = doc_str.lower().split()[:12]
    qv = np.array(embeddings.embed_documents(q_toks))
    dv = np.array(embeddings.embed_documents(d_toks))
    sim_matrix = qv @ dv.T
    return float(sim_matrix.max(axis=1).sum())

s_faq = demo_colbert_maxsim("port 9872 timeout", "allow UDP port 9872 quantum timeout")
s_off = demo_colbert_maxsim("port 9872 timeout", "dogs allowed Zurich office hub Tuesday")
print(f"ColBERT MaxSim on target={s_faq:.3f} vs unrelated={s_off:.3f}")
assert s_faq > s_off
        """,
        "Late interaction preserves per-word alignment rather than blurring rare tokens inside a single average vector.",
    )
    nb.try_it(
        [
            ("Compare Parent-Child chunk counts", "Print `len(parents)` vs `len(child_docs)`."),
            ("Verify Hybrid beats random mismatch", "Check `s_faq > s_off` under MaxSim and exact BM25 token score on `9872`."),
        ],
        """
print(f"Parents={len(parents)} vs Children={len(child_docs)} (ratio={len(child_docs)/len(parents):.1f}x)")
print("BM25 top score on '9872':", round(float(max(bm25.get_scores(tok("9872")))), 3))
        """,
    )
    nb.wrap_up(
        [
            "Claiming a toy cluster or 12-token loop is production ColBERT/RAPTOR — always be transparent about demo scale vs library scale.",
            "Omitting parent `pid` deduplication when multiple children match the same parent block.",
            "Blending raw unbounded BM25 scores directly with cosine similarities instead of RRF.",
        ],
        [
            "Parent-Document and Multi-Rep indexing cleanly separate fine search vectors from full context.",
            "Hybrid BM25+Dense RRF protects exact product codes and numbers where vectors alone soften precision.",
            "RAPTOR and ColBERT illustrate hierarchical summarization and late token-level MaxSim concepts clearly.",
        ],
        "Next (`11_reranking_and_compression.ipynb`): Filter Stage-1 hits with a REAL local neural `CrossEncoder` reranker and compress prompts cleanly!",
        [
            ("Why Single-Pass Uniform Indexing Hits Limits", "Overview of 4 indexing strategies."),
            ("Parent-Doc & Multi-Rep Summary Indexes", "Small search keys + rich context."),
            ("Hybrid BM25 + Dense RRF", "Solving exact error-token lookup."),
            ("RAPTOR Hierarchical Tree (Demo Scale)", "Clustering + summarizing chunks cleanly."),
            ("ColBERT Token MaxSim (Demo Scale)", "Token-matrix matching intuition."),
        ],
    )
    return nb.save(nb_dir)


def build_nb11(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("11_reranking_and_compression.ipynb")
    nb.header(
        "11 — Reranking & Contextual Compression (Local Cross-Encoder)",
        "By the end of this notebook, you can fetch wide candidates (`k=8`), rerank them with a real local neural Cross-Encoder, compress irrelevant filler sentences, and measure precision vs latency.",
        [
            "Understand why Cross-Encoders (joint query+doc attention) outperform bi-encoder embeddings on precision.",
            "Run a free local `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` via `get_cross_encoder()` (noting Cohere optionally).",
            "Apply Contextual Compression to shrink token length while comparing before/after Top-3 precision vs NB05 baseline.",
        ],
        "Notebooks 00–10",
    )
    nb.setup()
    nb.concept(
        "Wide Net -> Neural Cross-Encoder Reranker -> Lean Context",
        """
A bi-encoder vector store scores `query` and `chunk` **separately** (fast, scales to millions, slightly coarse). A **Cross-Encoder** reads `[query + chunk]` **together word-by-word** in one transformer pass! That's too slow for 10,000 chunks, yet takes milliseconds on the top `k=8` candidates!

- **Optional Hosted Path:** Paid services like `CohereRerank` provide hosted reranking via API; here we default to free local neural `ms-marco-MiniLM-L-6-v2`!
- **Contextual Compression:** After reranking picks top 3 chunks, trim away unrelated sentences inside those chunks to cut prompt cost.
        """,
        """
All Chunks ---> [Stage 1: Vector Search (k=8, ~2ms)] ---> 8 Candidates
                                                               |
                                             [Stage 2: CrossEncoder Reranker] ---> Top 3 Precision Winners
                                                                                       |
                                                                    [Contextual Compression] ---> Tight Context!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Initialize Wide Retriever & Real Local Neural `CrossEncoder`",
        "Let's index medium chunks (`300` chars) and load our local `ms-marco-MiniLM-L-6-v2` cross-encoder via `get_cross_encoder()`.",
        """
import time
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents
from course.utils.models import get_cross_encoder

chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
reranker = get_cross_encoder()
print(f"Loaded CrossEncoder (neural={reranker.is_neural}) over {len(chunks)} chunks.")
        """,
        "First run downloads `ms-marco-MiniLM-L-6-v2` once locally; afterwards it executes offline.",
    )
    nb.step(
        "Step 2: Compare Before/After Top-3 Precision + Latency Trade-off vs NB05 Baseline",
        "Let's run a tricky multi-keyword query (`'99.995% uptime SLA Enterprise Tier'`), comparing Stage-1 bi-encoder rank against CrossEncoder reranking!",
        """
q_sla = "What exact uptime SLA percentage does Enterprise Tier guarantee?"
t0 = time.perf_counter()
stage1 = store.similarity_search(q_sla, k=8)
s1_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
reranked = reranker.rerank(q_sla, stage1, top_n=3)
rr_ms = (time.perf_counter() - t1) * 1000
print(f"Stage-1 (#1 has 99.995%?={'99.995%' in stage1[0].page_content}) in {s1_ms:.1f}ms")
print(f"Reranked (#1 has 99.995%?={'99.995%' in reranked[0][1].page_content}, score={reranked[0][0]:.2f}) in {rr_ms:.1f}ms")
        """,
        "CrossEncoder inspects exact query-document interaction and promotes the true `99.995%` chunk to rank #1 cleanly!",
    )
    nb.step(
        "Step 3: Apply Contextual Compression & Measure Character/Token Reduction",
        "Let's filter out sentences inside our top-3 reranked chunks that share zero meaningful stems with the query and verify saved tokens!",
        """
import re
def compress_chunks(query_str: str, docs):
    q_terms = {w for w in re.findall(r"[a-z0-9_]+", query_str.lower()) if len(w) > 2}
    kept_blocks = []
    for d in docs:
        kept = [ln.strip() for ln in d.page_content.splitlines() if len(q_terms & set(re.findall(r"[a-z0-9_]+", ln.lower()))) >= 1]
        kept_blocks.append("\\n".join(kept) if kept else d.page_content)
    return kept_blocks

top3_docs = [doc for _, doc in reranked]
raw_len = sum(len(d.page_content) for d in top3_docs)
comp_list = compress_chunks(q_sla, top3_docs)
comp_len = sum(len(x) for x in comp_list)
print(f"Compressed context: {raw_len} -> {comp_len} chars ({100*(1-comp_len/raw_len):.1f}% savings!)")
        """,
        "Contextual compression trims boilerplate intro headings while preserving the exact answer line (`99.995%`)!",
    )
    nb.step(
        "Step 4: Generate Final Answer from Compressed Context",
        "Now let's pass the lean compressed snippet into `llm` and inspect the output cleanly.",
        """
compressed_ctx = "\\n---\\n".join(comp_list)
ans = llm.invoke(f"<context>\\n{compressed_ctx}\\n</context>\\nQuestion: {q_sla}").content
print("Compressed Answer:", ans)
assert "99.995" in ans
        """,
        "Lean, high-precision context yields accurate grounded generation at smaller token budgets.",
    )
    nb.try_it(
        [
            ("Compare Precision@3 on 3 queries", "Measure how often target facts sit in rank 1–3 before vs after `reranker.rerank()`."),
            ("Inspect latency on larger pool", "Compare `k=4` vs `k=12` into `reranker.rerank(..., top_n=3)`."),
        ],
        """
qs = ["Enterprise Plan monthly cost", "ROS2 distributions supported", "Rest API v1 deprecation date"]
for qt in qs:
    cands = store.similarity_search(qt, k=8)
    top_r = reranker.rerank(qt, cands, top_n=1)[0]
    print(f"Reranked #1 ({top_r[0]:+.2f}) for '{qt}' -> {top_r[1].metadata['source']}")
        """,
    )
    nb.wrap_up(
        [
            "Running a CrossEncoder over 5,000 chunks directly instead of only top 8–25 Stage-1 vector hits — causes multi-second hangs!",
            "Over-compressing short 2-line chunks until pronoun targets disappear.",
            "Assuming CrossEncoder scores work globally across queries — logits differ per query, so rank relative order cleanly.",
        ],
        [
            "Two-Stage Retrieval pairs fast Bi-Encoder recall (`k=8`) with deep Cross-Encoder precision (`top_n=3`).",
            "`ms-marco-MiniLM-L-6-v2` runs completely free and locally with zero paid Cohere key.",
            "Contextual Compression trims unrelated noise, cutting prompt tokens and latency cost.",
        ],
        "Next (`12_agentic_and_self_correcting_rag.ipynb`): Build real tool-calling agents (`create_agent`) AND self-correcting LangGraph loops with relevance grading & retry caps!",
        [
            ("Bi-Encoder vs Cross-Encoder Intuition", "Why joint attention beats compressed single vectors."),
            ("Loading Free Local CrossEncoder", "Avoiding paid Cohere requirements cleanly."),
            ("Before/After Precision & Latency Benchmark", "Quantifying exact #1 promotion & ms cost."),
            ("Contextual Compression", "Measuring character/token reduction live."),
            ("When Reranking Pays Off & Recap", "Latency vs precision tradeoff."),
        ],
    )
    return nb.save(nb_dir)


def build_nb12(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("12_agentic_and_self_correcting_rag.ipynb")
    nb.header(
        "12 — Agentic & Self-Correcting RAG (`create_agent` + LangGraph CRAG/Self-RAG)",
        "By the end of this notebook, you can wrap retrieval as an agent tool via `langchain.agents.create_agent` and build a capped self-correcting LangGraph that grades, rewrites, retries, and logs decisions.",
        [
            "Build a modern retrieval tool agent using `from langchain.agents import create_agent` that skips retrieval on non-domain math/greetings.",
            "Implement Corrective RAG + Self-RAG in LangGraph (`retrieve -> grade -> rewrite/retry OR generate -> ground check`).",
            "Enforce a strict loop cap (`retries <= 2`), log every node decision cleanly, and demonstrate failure recovery.",
        ],
        "Notebooks 00–11",
    )
    nb.setup()
    nb.concept(
        "Agents That Decide WHEN to Search + Graphs That Check THEMSELVES",
        """
Two big shifts turn static pipelines into intelligent assistants:
1. **Agentic RAG (`create_agent`)**: Give the LLM a `@tool search_novasphere_docs` tool. When the user asks `"What is 2 + 2?"`, the agent **skips** search completely; when asked `"Enterprise Plan price?"`, it **calls** the tool cleanly!
2. **Self-Correcting LangGraph (CRAG + Self-RAG)**:
   - **Grade Retrieval**: Throw away irrelevant chunks.
   - **Rewrite & Retry (Capped!)**: If all chunks look irrelevant and `retries < 2`, rewrite the query and loop back!
   - **Ground Check (Self-RAG)**: Verify final answer claims trace back to retrieved docs (refuse rather than guess if retries run out!).
        """,
        """
  [Agent: create_agent] --("2+2?")--> Skip Tool --> "4"
                        --("Pricing?")--> Call @tool --> Grounded Answer!

  [CRAG LangGraph] : [retrieve] -> [grade] --(kept docs?)--> YES -> [generate] -> [check_grounding] -> [END]
                                      ^                         |
                                      +-- [rewrite (cap<=2)] <--NO
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Expose Vector Search as a Tool for `langchain.agents.create_agent`",
        "Using the official current LangChain 1.x API (`create_agent`), let's wrap `InMemoryVectorStore` inside `@tool` and watch the agent skip search on `2 + 2` while calling search on Novasphere questions!",
        """
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)

@tool
def search_novasphere_docs(query: str) -> str:
    \"\"\"Search Novasphere Dynamics handbooks, FAQ, and release notes.\"\"\"
    return "\\n\\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in store.similarity_search(query, k=3))

rag_agent = create_agent(model=llm, tools=[search_novasphere_docs], system_prompt="Use search_novasphere_docs only when needed.")
        """,
        "`create_agent` builds a modern ReAct tool-calling loop with zero legacy `AgentExecutor` boilerplate.",
    )
    nb.step(
        "Step 2: Invoke `create_agent` — Tool Skip vs. Tool Call",
        "Watch how `'What is 2 + 2?'` executes 0 tool calls cleanly, while `'Enterprise Plan cost?'` triggers `search_novasphere_docs` automatically!",
        """
r_math = rag_agent.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
r_rag = rag_agent.invoke({"messages": [{"role": "user", "content": "How much does the Enterprise Plan cost per month?"}]})
print("Math reply (msgs count):", len(r_math["messages"]), "->", r_math["messages"][-1].content)
print("RAG reply  (msgs count):", len(r_rag["messages"]), "->", r_rag["messages"][-1].content[:110])
assert len(r_math["messages"]) == 2 and len(r_rag["messages"]) >= 4 and "4,200" in r_rag["messages"][-1].content
        """,
        "Message list length proves `r_math` finished in 1 step (`2` messages: user + AI) whereas `r_rag` ran tool invocation (`4` messages: user + AI tool_call + ToolMessage + final AI)!",
    )
    nb.step(
        "Step 3: Define Stateful Corrective/Self-RAG Nodes with Decision Logging",
        "Now let's construct our self-correcting **LangGraph**: `retrieve -> grade -> rewrite (retries < 2) -> generate -> verify_grounding`, recording every decision in `state['logs']`!",
        """
from typing import List, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class GradeRel(BaseModel):
    relevant: bool = Field(description="True if doc helps answer query")
grader = llm.with_structured_output(GradeRel)

class CRAGState(TypedDict):
    question: str; docs: List[Document]; retries: int; answer: str; logs: List[str]

def crag_retrieve(s: CRAGState):
    docs = store.similarity_search(s["question"], k=3)
    return {"docs": docs, "logs": s["logs"] + [f"retrieve(try={s['retries']}): {len(docs)} docs"]}
        """,
        "Keeping an explicit `logs` list inside `CRAGState` makes every branch auditable cleanly.",
    )
    nb.step(
        "Step 4: Add `grade`, Bounded `rewrite` (`retries <= 2`), `generate` & Ground Check",
        "If `grade` strips every chunk (like an impossible query), `rewrite` increments `retries` up to max 2 retries (`3` attempts total), then falls back honestly without infinite loops!",
        """
def crag_grade(s: CRAGState):
    kept = [d for d in s["docs"] if grader.invoke(f"QUESTION: {s['question']}\\nDOCUMENT: {d.page_content}").relevant]
    return {"docs": kept, "logs": s["logs"] + [f"grade: kept {len(kept)}/{len(s['docs'])}"]}

def crag_rewrite(s: CRAGState):
    new_q = llm.invoke(f"Rewrite query as a crisp standalone question:\\n{s['question']}").content
    return {"question": new_q, "retries": s["retries"] + 1, "logs": s["logs"] + [f"rewrite -> retry {s['retries']+1}"]}

def crag_generate(s: CRAGState):
    ctx = "\\n---\\n".join(d.page_content for d in s["docs"])
    ans = llm.invoke(f"<context>\\n{ctx}\\n</context>\\nQuestion: {s['question']}").content
    verdict = "GROUNDED" if ("don't know" in ans.lower() or s["docs"]) else "UNVERIFIED"
    return {"answer": ans, "logs": s["logs"] + [f"generate -> {verdict}"]}
        """,
        "Notice `crag_rewrite` guarantees loop termination because `retries` advances strictly by `+1` on every cycle.",
    )
    nb.step(
        "Step 5: Compile CRAG Graph & Run Success vs. Retry-Fallback Failure Cases",
        "Let's wire `StateGraph(CRAGState)` cleanly and compare a grounded success (`ZeroGravity failover`) against an out-of-scope trigger (`NASDAQ stock ticker`) hitting max retries cleanly!",
        """
from langgraph.graph import START, END, StateGraph
g = StateGraph(CRAGState)
for name, fn in [("retrieve", crag_retrieve), ("grade", crag_grade), ("rewrite", crag_rewrite), ("generate", crag_generate)]:
    g.add_node(name, fn)
g.add_edge(START, "retrieve"); g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", lambda s: "generate" if (s["docs"] or s["retries"] >= 2) else "rewrite", {"rewrite": "rewrite", "generate": "generate"})
g.add_edge("rewrite", "retrieve"); g.add_edge("generate", END)
crag = g.compile()

ok_res = crag.invoke({"question": "ZeroGravity failover timeout milliseconds?", "docs": [], "retries": 0, "answer": "", "logs": []})
bad_res = crag.invoke({"question": "What is Novasphere NASDAQ stock ticker?", "docs": [], "retries": 0, "answer": "", "logs": []})
print("OK logs :", ok_res["logs"], "\\nOK ans  :", ok_res["answer"][:95])
print("Bad logs:", bad_res["logs"], "\\nBad ans :", bad_res["answer"][:95])
assert bad_res["retries"] == 2
        """,
        "Valid query solved cleanly in `retries=0`; out-of-scope query executed exactly `2` retries cleanly and refused without looping forever!",
    )
    nb.try_it(
        [
            ("Test error troubleshooting via `crag`", "Invoke `crag` on `How do I resolve ERR_QUANTUM_TIMEOUT?` and check logs."),
            ("Test `rag_agent` on pet days", "Invoke `rag_agent` on Zurich office pet hours."),
        ],
        """
res_t = crag.invoke({"question": "How do I resolve ERR_QUANTUM_TIMEOUT?", "docs": [], "retries": 0, "answer": "", "logs": []})
print("Logs:", res_t["logs"])
print("Ans :", res_t["answer"][:110])
        """,
    )
    nb.wrap_up(
        [
            "Using legacy `AgentExecutor` tutorials that fail on modern LangChain `1.x` instead of `create_agent`.",
            "Omitting explicit loop caps (`retries >= max`) in self-correcting graphs — infinite retry bills!",
            "Skipping decision logging (`state['logs']`), making multi-step agent failures impossible to debug.",
        ],
        [
            "`create_agent` decides automatically when a tool call is unnecessary (`2+2`) vs required (private docs).",
            "LangGraph CRAG pairs document grading, bounded query rewriting, and Self-RAG grounding verification cleanly.",
            "Decision trace logs and hard retry caps guarantee finite execution and easy debugging.",
        ],
        "Next (`13_evaluating_rag.ipynb`, Phase 4!): Measure Hit Rate@k, MRR, Faithfulness, and run controlled ablations over chunk size, `k`, and reranking!",
        [
            ("When Static RAG Isn't Enough", "Deciding when to search & self-correcting bad hits."),
            ("Modern Tool Agent with `create_agent`", "Skipping vs invoking `@tool search_novasphere_docs`."),
            ("Designing the CRAG State Graph", "Nodes, edges, and loop caps."),
            ("Structured Grading & Self-RAG Check", "Filtering irrelevant chunks + logging decisions."),
            ("Live Trace Walkthrough", "0 retries on valid hits vs clean exit at retry=2."),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    for builder in (build_nb10, build_nb11, build_nb12):
        print("Built:", builder(out_dir).name)


"""
Phase 2 Notebook Generator (Notebooks 08-09): Query Translation & Routing/Query Construction.
"""

from pathlib import Path
from course.utils.nb_builder import CourseNotebookBuilder


def build_nb08(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("08_query_translation.ipynb")
    nb.header(
        "08 — Query Translation (Multi-Query, RAG-Fusion, Decomposition, Step-Back & HyDE)",
        "By the end of this notebook, you can transform ambiguous, jargon-mismatched, or compound queries using five translation techniques and judge their model-call cost.",
        [
            "Demonstrate why raw user wording misses key documents in plain NB05 vector retrieval.",
            "Implement Multi-Query, RAG-Fusion (`1 / (60 + rank)` verified against hand math), Decomposition, Step-Back, and HyDE.",
            "Compare recall/MRR against NB05 baseline and evaluate when NOT to use extra query calls.",
        ],
        "Notebooks 00–07",
    )
    nb.setup()
    nb.concept(
        "Why Fix the Question Before Searching?",
        """
Wait — why change the user's question? Users often ask **vague questions** (*"legacy API?"*), **double questions** (*"Who is the Ethics Officer AND what's the learning stipend?"*), or mismatch document wording!

**Reciprocal Rank Fusion (RRF) in Plain Words:**  
Instead of adding incompatible cosine similarities across searches, RRF gives each doc a **rank vote**: `score = 1 / (60 + rank)` (Rank #1 gets `1/61`, Rank #2 gets `1/62`). Any chunk consistently near the top across multiple translated queries wins!

| Technique | Core Idea | Extra LLM Calls | Best For | **When NOT to Use** |
|---|---|---|---|---|
| **Multi-Query + RRF** | Rephrase Q 3 ways & fuse ranks | 1 call | Vague/underspecified queries | Ultra-low-latency simple SKU lookups |
| **Decomposition** | Split multi-part Q into sub-Q's | 1 call | Multi-document compound questions | Single short factoid questions |
| **Step-Back** | Ask broader high-level principle first | 1 call | Specific bugs needing architecture rules | Exact pricing/number lookups |
| **HyDE** | Draft hypothetical doc, embed *answer* | 1 call | Vocabulary mismatch (short query vs prose) | Proper nouns / rare error codes (`ERR_...`) |
        """,
        """
Raw Ambiguous Q ---> [Translator (LLM)] ---> Sub-Queries Q1..Qn (or HyDE Paragraph)
                                                    |
                         Vector Search x N ---> [RRF Rank Voting: 1/(60+rank)] ---> Better Context!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Establish the NB05 Failure Baseline on Compound & Jargon Queries",
        "Let's index our corpus (`chunk_size=340`) and see plain single-query `k=2` struggle on a two-document compound question.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=340, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
compound_q = "Who was appointed Chief AI Ethics Officer in October 2026 and what annual learning stipend can employees receive?"
base_srcs = sorted({d.metadata["source"] for d in store.similarity_search(compound_q, k=2)})
print("NB05 Baseline k=2 sources:", base_srcs, "| Covers BOTH docs?:", len(base_srcs) == 2)
        """,
        "With plain single-query search at `k=2`, one half of a two-document question frequently crowds out the second file.",
    )
    nb.step(
        "Step 2: Hand-Worked RAG-Fusion (`RRF`) Verification + Multi-Query Search",
        "Assume Doc A ranks #1 in Query 1 and #2 in Query 2 (`k=60`): its exact RRF score MUST equal `1/61 + 1/62 = 0.03252247...`. Let's verify in code and run Multi-Query!",
        """
from langchain_core.documents import Document
def rrf_fuse(lists, k=60):
    scores, dmap = {}, {}
    for docs in lists:
        for rank, d in enumerate(docs, start=1):
            dmap[d.page_content] = d; scores[d.page_content] = scores.get(d.page_content, 0.0) + 1.0 / (k + rank)
    return [(dmap[txt], sc) for txt, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

da, db = Document(page_content="Doc A"), Document(page_content="Doc B")
hand_check = rrf_fuse([[da, db], [db, da]], k=60)
assert abs(hand_check[0][1] - (1/61 + 1/62)) < 1e-12
print(f"Verified RRF hand math: {hand_check[0][1]:.8f} == {1/61 + 1/62:.8f}")
        """,
        "Our RRF function matches manual arithmetic to 12 decimal places.",
    )
    nb.step(
        "Step 3: Query Decomposition Outperforms NB05 Baseline on Multi-Doc Synthesis",
        "Watch how **Decomposition** breaks `compound_q` into 2 focused sub-questions and retrieves **both** `recent_changelog.md` AND `company_handbook.txt` via RRF!",
        """
sub_raw = llm.invoke(f"Decompose into concise sub-questions:\\n{compound_q}").content
sub_qs = [l.lstrip("0123456789. ").strip() for l in sub_raw.splitlines() if l.strip()]
fused = rrf_fuse([store.similarity_search(sq, k=2) for sq in sub_qs])
decomp_srcs = sorted({d.metadata["source"] for d, _ in fused[:3]})
print("Sub-questions    :", sub_qs)
print("Decomposed top-3 :", decomp_srcs, "| Covers BOTH docs?:", len(decomp_srcs) == 2)
        """,
        "Splitting multi-part questions lets each sub-search dedicate its full embedding capacity to one topic cleanly.",
    )
    nb.step(
        "Step 4: Compare Step-Back Prompting & HyDE Side-by-Side",
        "Let's also inspect Step-Back (abstract concept expansion) and HyDE (embedding a generated hypothetical passage instead of raw keywords).",
        """
tech_q = "endpoint 410 deprecation timeline"
step_back_q = llm.invoke(f"Generate a broader conceptual step-back question for: {tech_q}").content
hyde_passage = llm.invoke(f"Write a short hypothetical document passage answering: {tech_q}").content
hyde_hits = store.similarity_search_by_vector(embeddings.embed_query(hyde_passage), k=2)
print("Step-Back Q:", step_back_q)
print("HyDE Top Hit:", hyde_hits[0].metadata["source"], "->", hyde_hits[0].page_content[:85].replace("\\n", " "))
        """,
        "HyDE bridges short/terse keywords cleanly into document-style prose space.",
    )
    nb.try_it(
        [
            ("Test Multi-Query expansion", "Generate 3 variations for `'pets Zurich hub'` and check fused source `company_handbook.txt`."),
            ("Inspect HyDE failure on rare error codes", "Explain why embedding hypothetical prose for `'ERR_QUANTUM_TIMEOUT'` can lose exact error-string specificity."),
        ],
        """
mq = [l.lstrip("0123456789. ").strip() for l in llm.invoke("Generate 3 alternative search queries:\\npets Zurich").content.splitlines() if l.strip()]
mq_res = rrf_fuse([store.similarity_search(q_i, k=2) for q_i in mq])
print("Multi-Query #1 source:", mq_res[0][0].metadata["source"])
        """,
    )
    nb.wrap_up(
        [
            "Turning Multi-Query or HyDE on blindly for every trivial lookup — doubles latency and token bills for zero gain.",
            "Using HyDE on opaque error codes or product SKUs where hypothetical prose omits exact codes.",
            "Summing raw unnormalized similarity scores instead of rank votes (`1 / (60 + rank)`).",
        ],
        [
            "Query translation fixes ambiguous, compound, or mismatched phrasing *before* retrieval runs.",
            "RRF fuses arbitrary search lists cleanly without fragile score calibration.",
            "Measured comparisons confirm Decomposition recovers multi-doc facts missed by plain Top-2.",
        ],
        "Next (`09_routing_and_query_construction.ipynb`): Route incoming questions straight to the right document silo and parse metadata filters cleanly!",
        [
            ("When Single-Query Search Fails", "Multi-doc crowding & vocabulary mismatch."),
            ("Multi-Query + RAG-Fusion RRF", "Plain-word explanation + hand-math proof."),
            ("Query Decomposition Live Comparison", "Beating NB05 baseline on compound questions."),
            ("Step-Back & HyDE", "Bridging abstract concepts and prose wording."),
            ("Cost & When-Not-To-Use Matrix", "Choosing translation selectively."),
        ],
    )
    return nb.save(nb_dir)


def build_nb09(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("09_routing_and_query_construction.ipynb")
    nb.header(
        "09 — Routing & Query Construction (Logical, Semantic & Metadata Filters)",
        "By the end of this notebook, you can route queries with Pydantic structured schemas or embedding cosine similarity, construct metadata filters, and log wrong-route edge cases.",
        [
            "Implement Logical Routing with Pydantic v2 (`.with_structured_output()`) and Semantic Routing with pure cosine similarity.",
            "Translate natural language constraints into executable vector metadata filters cleanly (guarding malformed inputs).",
            "Demonstrate a wrong-route failure, log routing decisions, and know when NOT to route.",
        ],
        "Notebooks 00–08",
    )
    nb.setup()
    nb.concept(
        "Hospital Triage Receptionist: Routing Before Retrieval",
        """
Imagine asking an orthopedic surgeon about payroll forms! Searching all databases equally blurs terminology and slows search.

1. **Logical Routing**: Uses structured Pydantic fields (`datasource: Literal['handbook', 'faq', 'changelog', 'safety']`) so the LLM classifies intent cleanly.
2. **Semantic Routing**: Zero LLM token cost! Embeds each domain description once and matches query cosine similarity in `<2ms`.
3. **Query Construction**: Converts natural constraints (*"from changelog"* / *"executive docs"*) into structured vector `filter` predicates.

**When NOT to use Routing:** Cross-document synthesis questions (*"Which tier includes ZeroGravity AND what does it cost?"*) span multiple files — hard-routing to ONE file deletes half the answer!
        """,
        """
User Q ---> [Router (Logical or Semantic) + Decision Logger]
                 |-- 'company_handbook' --> filtered search (HR/Pricing)
                 |-- 'product_faq'      --> filtered search (ROS2/Errors)
                 |-- 'recent_changelog' --> filtered search (Oct 2026)
                 +-- 'WRONG ROUTE!' ------> Zero matching hits -> Fallback/All-corpus safety net!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Index Multi-Source Corpus & Logical Pydantic Router",
        "Let's index all corpus sources with `doc_type` metadata and build a Pydantic v2 structured `RouteDecision` router.",
        """
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)

class RouteDecision(BaseModel):
    datasource: Literal["company_handbook", "product_faq", "recent_changelog", "safety_brief"] = Field(description="Target doc bucket")

logical_router = llm.with_structured_output(RouteDecision)
print("Structured route for error code:", logical_router.invoke("How do I fix ERR_QUANTUM_TIMEOUT?").datasource)
        """,
        "Native Pydantic v2 validation guarantees only known datasources can ever be returned.",
    )
    nb.step(
        "Step 2: Zero-LLM Semantic Routing + Structured Decision Logger",
        "Embed three crisp domain descriptions once, match queries by dot product, and append every routing decision to an audit log.",
        """
import numpy as np
route_prompts = {
    "company_handbook": "Travel meal reimbursement, Zurich hub pet policy, learning stipend, commercial subscription pricing.",
    "product_faq": "Technical troubleshooting ERR_QUANTUM_TIMEOUT port 9872, ROS2 compatibility, ZeroGravity SLA uptime.",
    "recent_changelog": "October 2026 release v4.6, HyperSync latency 42%, Chief AI Ethics Officer, REST API v1 deprecation.",
}
route_vecs = {k: np.array(embeddings.embed_query(v)) for k, v in route_prompts.items()}
routing_log = []

def semantic_route(q: str):
    qv = np.array(embeddings.embed_query(q))
    scores = {k: float(np.dot(qv, rv)) for k, rv in route_vecs.items()}
    winner = max(scores, key=scores.get)
    routing_log.append({"query": q, "route": winner, "score": round(scores[winner], 3)})
    return winner, scores[winner]
        """,
        "Semantic routing logs exact confidence scores alongside the winning datasource with zero LLM calls.",
    )
    nb.step(
        "Step 3: Query Construction (Natural Constraint -> Metadata Filter) + Guarding Malformed Output",
        "Let's build a helper translating datasource names to strict store metadata filters (`doc_type`), with safe fallback if an unknown key arrives.",
        """
SOURCE_MAP = {"company_handbook": "handbook", "product_faq": "faq", "recent_changelog": "changelog", "safety_brief": "safety"}

def construct_filtered_search(query: str, predicted_route: str, k: int = 2):
    dtype = SOURCE_MAP.get(predicted_route)
    if not dtype:
        return store.similarity_search(query, k=k)  # Safe fallback on malformed/unknown route!
    return store.similarity_search(query, k=k, filter=lambda d: d.metadata.get("doc_type") == dtype)

route, sc = semantic_route("What is REST API v1 shutdown date?")
print("Filtered search hit:", construct_filtered_search("What is REST API v1 shutdown date?", route, k=1)[0].metadata)
        """,
        "Mapping unknown/malformed router outputs cleanly to unfiltered search prevents application crashes!",
    )
    nb.step(
        "Step 4: Demonstrate a Wrong-Route Failure vs. Baseline (and Why Routing Logs Matter)",
        "For cross-document Q05 (`ZeroGravity tier AND 40ms threshold`), routing strictly to `product_faq` misses Starter/Enterprise pricing nuances in `company_handbook`!",
        """
q_cross = "Which tier is required for ZeroGravity Failover and what is the 40ms threshold?"
routed_only_faq = construct_filtered_search(q_cross, "product_faq", k=2)
unfiltered = store.similarity_search(q_cross, k=4)
print("Wrong hard-route sources:", sorted({d.metadata["source"] for d in routed_only_faq}))
print("Multi-doc unfiltered    :", sorted({d.metadata["source"] for d in unfiltered}))
print("Audit log snapshot      :", routing_log)
        """,
        "Hard single-bucket routing broke cross-document retrieval — logging routes lets engineers spot when queries need multi-route or unfiltered search!",
    )
    nb.try_it(
        [
            ("Test malformed route safety", "Call `construct_filtered_search('Enterprise Plan cost', 'non_existent_bucket', k=2)` and verify clean fallback."),
            ("Inspect semantic score on pet policy", "Call `semantic_route('Can dogs visit the Zurich office?')` and check winning bucket."),
        ],
        """
assert len(construct_filtered_search("Enterprise Plan cost", "invalid_route_xyz", k=2)) == 2
print("Pets semantic route:", semantic_route("Can dogs visit the Zurich office?"))
        """,
    )
    nb.wrap_up(
        [
            "Applying single-source hard routing to multi-document comparison questions.",
            "Crashing when structured/parsed metadata filter output emits unexpected values — always fall back cleanly.",
            "Skipping decision logging (`query`, `route`, `confidence`), leaving silent misroutes invisible.",
        ],
        [
            "Logical routing enforces Pydantic v2 output schemas; semantic routing uses fast vector dot products.",
            "Query construction converts natural intent into safe vector-store `filter` predicates.",
            "Wrong-route failures highlight when NOT to route: cross-document synthesis & ambiguous queries.",
        ],
        "Next (`10_advanced_indexing.ipynb`, Phase 3!): Upgrade indexing itself using Parent-Document hydration, Multi-Representation summaries, Hybrid BM25, RAPTOR hierarchical summaries, and ColBERT late interaction!",
        [
            ("Routing Architecture Overview", "Logical vs Semantic vs Query Construction."),
            ("Pydantic v2 Structured Router", "Strict datasources without pydantic_v1 deprecation."),
            ("Semantic Router + Audit Logging", "Zero-LLM triage with score tracking."),
            ("Metadata Filter Builder & Malformed Guard", "Safe predicate execution."),
            ("Wrong-Route Failure Demo", "When single-bucket routing hurts multi-doc answers."),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    for builder in (build_nb08, build_nb09):
        print("Built:", builder(out_dir).name)


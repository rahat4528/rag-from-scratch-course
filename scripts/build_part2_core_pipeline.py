"""
Builder for Part 2 (Notebooks 04-06): Embeddings, Vector Stores, Retrieval & Grounded Generation.
"""

from pathlib import Path
from scripts.nb_builder import NotebookBuilder


def build_nb04(nb_dir: Path) -> Path:
    nb = NotebookBuilder("04_embeddings_and_vector_stores.ipynb")
    nb.header(
        "04 — Embeddings & Vector Stores (Meaning as Coordinates)",
        "By the end of this notebook, you can convert text into numeric embeddings, measure cosine similarity, visualize 2D meaning maps with PCA, and persist a vector store.",
        [
            "Understand what an embedding vector is using a map-coordinates analogy.",
            "Measure semantic closeness with dot-product / cosine similarity and inspect 2D PCA coordinates.",
            "Index real chunks in `InMemoryVectorStore`, dump to JSON on disk, and reload with identical results.",
        ],
        "Notebooks 00–03",
    )
    nb.setup_section()
    nb.concept_section(
        "Words to Coordinates on a Meaning Map",
        """
Computers cannot compare the *meaning* of two English sentences with `==`. Instead, an **Embedding Model** acts like a master cartographer that places every phrase onto a multi-dimensional map (for example, 384 coordinates).

- Texts with related meanings (e.g., *"Enterprise pricing plan"* and *"USD 4,200 monthly subscription"*) land close together.
- Unrelated texts (*"chocolate cake recipe"*) land far away!
- **Cosine Similarity** measures how tightly two meaning vectors point in the same direction (`1.0` = same meaning, `0.0` = unrelated).
- A **Vector Store** acts as an indexed phonebook of coordinates so queries find top matches in milliseconds.
        """,
        """
  "Pets on Tuesday"  --> [ 0.21, -0.48, ..., 0.77 ]  \\
  "Dogs in Zurich"   --> [ 0.19, -0.44, ..., 0.75 ]  --> High Cosine Similarity (~0.7+)
  "ROS2 Linux boot"  --> [-0.62,  0.51, ..., 0.04 ]  --> Low Cosine Similarity  (~0.1)
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Embed Four Short Sentences and Inspect Vector Dimensions",
        "Let's embed four short sentences using our local embeddings helper and check vector shape and determinism.",
        """
import numpy as np
phrases = [
    "What is the monthly cost of the Enterprise tier?",
    "Enterprise Plan pricing is $4,200 per month with 200 nodes.",
    "Dogs are permitted in the Zurich hub on Tuesdays and Thursdays.",
    "Freshly baked chocolate chip cookies require butter and sugar.",
]
vecs = embeddings.embed_documents(phrases)
print(f"Embedded {len(vecs)} texts -> each vector has {len(vecs[0])} coordinates.")
        """,
        "Every text—whether 5 words or a paragraph—maps to the exact same 384-dimensional vector space.",
    )
    nb.step(
        "Step 2: Compute Cosine Similarity Between Query and Candidates",
        "Since normalized vectors have length 1.0, dot product equals cosine similarity. Let's compare `phrases[0]` against all four candidates.",
        """
q_vec = np.array(embeddings.embed_query(phrases[0]))
for p, v in zip(phrases, vecs):
    sim = float(np.dot(q_vec, np.array(v)) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
    print(f"sim={sim:+.3f} | {p}")
        """,
        "Notice how pricing question (`phrases[0]`) aligns highest with pricing answer (`phrases[1]`) and far lower with cookies (`phrases[3]`).",
    )
    nb.step(
        "Step 3: Compress 384 Dimensions Down to 2D with PCA for Visual Intuition",
        "Humans cannot visualize 384 axes, so let's project our 4 vectors to 2D `(x, y)` coordinates using scikit-learn's `PCA`.",
        """
from sklearn.decomposition import PCA
coords_2d = PCA(n_components=2).fit_transform(np.array(vecs))
for p, (x, y) in zip(phrases, coords_2d):
    print(f"({x:+.3f}, {y:+.3f}) <- {p[:48]}")
        """,
        "PCA gives you concrete x/y coordinates illustrating how related ideas cluster together on the 2D plane.",
    )
    nb.step(
        "Step 4: Build, Query, Save, and Reload an `InMemoryVectorStore`",
        "Let's index the real Novasphere chunks into `InMemoryVectorStore`, persist to JSON with `.dump()`, reload cleanly with `.load()`, and verify identical answers.",
        """
import tempfile
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=360, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
with tempfile.TemporaryDirectory() as tmp:
    dump_path = str(Path(tmp) / "store.json")
    store.dump(dump_path)
    reloaded = InMemoryVectorStore.load(dump_path, embeddings)
print("Reloaded match:", reloaded.similarity_search("ZeroGravity timeout threshold", k=1)[0].metadata)
        """,
        "Persisting your index with `.dump()` and `.load()` avoids re-embedding your entire library every time your app restarts.",
    )
    nb.try_it_yourself(
        [
            ("Test empty input safety", "Verify `embeddings.embed_query('')` returns a clean list of floats without throwing an error."),
            ("Search k larger than corpus", "Query a 2-doc store with `k=10` and verify it returns all available docs cleanly."),
        ],
        """
empty_vec = embeddings.embed_query("")
assert len(empty_vec) == 384
tiny_store = InMemoryVectorStore.from_texts(["Alpha telemetry", "Beta safety"], embeddings)
print("Oversized k returned:", len(tiny_store.similarity_search("telemetry", k=10)), "docs cleanly")
        """,
    )
    nb.wrap_up(
        [
            "Mixing two different embedding models (e.g., indexing with `MiniLM` and querying with `OpenAI`) — vectors live in completely incompatible spaces!",
            "Forgetting L2 normalization when writing manual dot-product similarity formulas.",
            "Expecting exact string equality (`sim == 1.0`) when searching paraphrased queries.",
        ],
        [
            "Embeddings encode semantic meaning as fixed-length numeric arrays (vectors).",
            "Cosine similarity ranks closeness by angle between vectors rather than exact keyword overlap alone.",
            "`InMemoryVectorStore` lets you build, query, dump, and reload vector indexes natively in pure Python.",
        ],
        "Next up (`05_retrieval_and_similarity.ipynb`): Master Top-k tuning, similarity score cutoffs, and MMR diversity search so you don't retrieve 3 copies of the same paragraph!",
        [
            ("The Meaning Map Intuition", "How coordinates and angles encode semantic similarity."),
            ("Embedding Real Sentences", "Vector dimensions and shapes in practice."),
            ("Visualizing Meaning with PCA", "Compressing 384D vectors to intuitive 2D points."),
            ("Vector Store Persistence", "Indexing, dumping JSON to disk, and reloading reliably."),
            ("Recap & Next Steps", "Avoiding mismatched embedding spaces."),
        ],
    )
    return nb.save(nb_dir)


def build_nb05(nb_dir: Path) -> Path:
    nb = NotebookBuilder("05_retrieval_and_similarity.ipynb")
    nb.header(
        "05 — Retrieval & Similarity Search (Top-k, Scores & MMR Diversity)",
        "By the end of this notebook, you can inspect similarity scores, filter irrelevant hits, and use MMR retrieval to return diverse context without duplicate chunks.",
        [
            "Use `similarity_search_with_score` to inspect confidence values and reject low-relevance garbage.",
            "Compare plain similarity search against Maximal Marginal Relevance (`search_type='mmr'`) for multi-topic diversity.",
            "Apply retrieval-time metadata filters to restrict search by source file or department.",
        ],
        "Notebook 04 (`04_embeddings_and_vector_stores.ipynb`)",
    )
    nb.setup_section()
    nb.concept_section(
        "Similarity Search vs. MMR Diversity",
        """
Plain **Top-k Similarity Search** pulls the *k* chunks closest to your question — which often means grabbing 3 slightly shifted overlapping windows from the exact same paragraph!

**Maximal Marginal Relevance (MMR)** solves redundancy like hiring a diverse project team:
1. Fetch a broader pool (`fetch_k=10` candidates).
2. Pick the single best candidate first.
3. For each remaining slot up to `k`, reward chunks that are **relevant to the query** YET **different from chunks already selected** (`lambda_mult` balances relevance vs diversity).
        """,
        """
Standard Top-3:  [ Chunk 1 (handbook sec 4), Chunk 2 (handbook sec 4 overlap), Chunk 3 (handbook sec 4) ]
MMR (Top-3)   :  [ Chunk 1 (handbook sec 4), Chunk A (product_faq ZeroGravity), Chunk X (changelog note) ]
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Index the Full Novasphere Corpus",
        "Let's load all four Novasphere files (including `policy_brief.pdf`), split cleanly, and populate an `InMemoryVectorStore`.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=340, chunk_overlap=45).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Indexed {len(chunks)} chunks across handbook, FAQ, changelog, and PDF brief.")
        """,
        "All documents are now indexed together with `source` and `doc_type` metadata preserved.",
    )
    nb.step(
        "Step 2: Inspect Raw Similarity Scores (Relevant vs. Off-Topic Query)",
        "Let's run `similarity_search_with_score` on an on-topic query (`Enterprise Plan cost`) vs an off-topic query (`banana smoothie recipe`) to see how confidence shifts.",
        """
for q in ["How much does the Enterprise Plan cost per month?", "How do you blend a banana strawberry smoothie?"]:
    scored = store.similarity_search_with_score(q, k=2)
    top_doc, top_score = scored[0]
    print(f"Query: {q[:44]:<44} | top_score={top_score:.3f} | src={top_doc.metadata['source']}")
        """,
        "On-topic questions score substantially higher than off-topic noise—giving us a threshold signal!",
    )
    nb.step(
        "Step 3: Compare Standard Similarity Search vs. MMR Diversity",
        "When a broad question touches multiple tiers or files, watch how MMR pulls a wider variety of document sources than plain Top-k.",
        """
broad_q = "What plans include ZeroGravity failover, HyperSync telemetry, and ROS2 support?"
sim_sources = [d.metadata["source"] for d in store.similarity_search(broad_q, k=4)]
mmr_sources = [d.metadata["source"] for d in store.max_marginal_relevance_search(broad_q, k=4, fetch_k=12, lambda_mult=0.5)]
print("Plain Top-4 sources:", sim_sources)
print("MMR Top-4 sources  :", mmr_sources)
        """,
        "MMR reduces repetition and spreads coverage across `company_handbook.txt`, `product_faq.md`, and `recent_changelog.md`.",
    )
    nb.step(
        "Step 4: Retrieve with Clean Metadata Filters",
        "Sometimes users only want recent updates (`recent_changelog.md`) or HR rules (`company_handbook.txt`). Passing a `filter` function narrows search instantly.",
        """
faq_only = store.similarity_search(
    "What is the failover threshold?",
    k=2,
    filter=lambda doc: doc.metadata.get("source") == "product_faq.md",
)
for d in faq_only:
    print(f"[{d.metadata['source']}] {d.page_content[:110].replace(chr(10), ' ')}...")
        """,
        "Metadata filtering prevents old manuals or wrong departments from ever polluting the retrieved context.",
    )
    nb.try_it_yourself(
        [
            ("Tune `lambda_mult` in MMR", "Compare `lambda_mult=0.1` (extreme diversity) vs `lambda_mult=0.9` (near pure similarity)."),
            ("Implement a threshold helper", "Filter `similarity_search_with_score(q, k=4)` to keep only chunks scoring >= 0.25."),
        ],
        """
q_test = "ROS2 distributions supported"
hits_with_scores = store.similarity_search_with_score(q_test, k=4)
filtered = [d for d, s in hits_with_scores if s >= 0.25]
print(f"Kept {len(filtered)}/{len(hits_with_scores)} hits above score threshold 0.25.")
        """,
    )
    nb.wrap_up(
        [
            "Assuming higher `k` (like `k=20`) is always better — flooding prompts with marginally related text dilutes key facts.",
            "Ignoring score thresholds when users ask out-of-scope questions, forcing unrelated chunks into the LLM.",
            "Setting `fetch_k < k` in MMR — always make `fetch_k` 2x–4x larger than `k` so diversity has candidates to pick from.",
        ],
        [
            "`similarity_search_with_score` lets you inspect retrieval confidence before generating an answer.",
            "MMR balances relevance and diversity so multi-part questions cover distinct source chunks.",
            "Metadata filters guarantee strict scoping by document type, date, or department.",
        ],
        "Next up (`06_generation_and_prompting.ipynb`): Wire retrievers to grounded prompts with LangChain LCEL (`|`) chains and teach the model to refuse when evidence is missing!",
        [
            ("Why Raw Top-k Repeats Itself", "The overlapping chunk problem explained."),
            ("Similarity Scores & Confidence", "Separating real matches from off-topic queries."),
            ("MMR Retrieval Deep Dive", "How `fetch_k` and `lambda_mult` diversify context."),
            ("Pre-Filtering with Metadata", "Zero-noise scoping per file or doc type."),
            ("Hands-On Lab & Recap", "Building a threshold-guarded retriever."),
        ],
    )
    return nb.save(nb_dir)


def build_nb06(nb_dir: Path) -> Path:
    nb = NotebookBuilder("06_generation_and_prompting.ipynb")
    nb.header(
        "06 — Generation, Grounding, Citations & LCEL Chains",
        "By the end of this notebook, you can assemble clean LangChain LCEL (`|`) pipelines that cite sources strictly from context and honestly say 'I don't know' on gaps.",
        [
            "Design defensive RAG system prompts that constrain answers strictly to retrieved evidence.",
            "Build composable pipelines using LangChain Expression Language (`RunnablePassthrough`, `ChatPromptTemplate`, `StrOutputParser`).",
            "Verify grounded answers include source citations and refuse unanswerable questions without fabricating facts.",
        ],
        "Notebooks 00–05",
    )
    nb.setup_section()
    nb.concept_section(
        "Grounded Prompting & LCEL Pipes",
        """
Think of **LCEL (LangChain Expression Language)** like Unix command-line pipes (`|`) or a factory assembly line: output from step 1 flows cleanly into step 2!

```
question ---> { "context": retriever | format_docs, "question": pass_through } ---> prompt ---> llm ---> clean string
```

Two rules make a production prompt trustworthy:
1. **Strict Grounding Guardrail**: Explicitly tell the LLM to ignore outside training memory and answer *only* from `<context>`.
2. **Explicit Refusal Clause**: Instruct the LLM: *"If the answer is not in the context, reply: I don't know based on the provided context."*
        """,
        """
User Question --+--> Retriever --> format_docs (adds [source] tags) --+
                |                                                     v
                +--> RunnablePassthrough() -------------------> ChatPromptTemplate --> LLM --> Cited String
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Prepare Vector Store & Format-Docs Helper with Source Labels",
        "Let's create our vector store and a tiny `format_docs` function that prepends `[filename]` onto every retrieved chunk so citations stay attached.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=360, chunk_overlap=40).split_documents(load_corpus_documents(data_dir))
retriever = InMemoryVectorStore.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\\n\\n".join(f"[{d.metadata.get('source', 'unknown')}]\\n{d.page_content}" for d in docs)
        """,
        "Now every text block inside `<context>` visibly carries its origin filename.",
    )
    nb.step(
        "Step 2: Craft a Strict Grounded Prompt Template",
        "We wrap retrieved context cleanly inside `<context>...</context>` XML delimiters so instructions never blur together with source text.",
        """
from langchain_core.prompts import ChatPromptTemplate
rag_prompt = ChatPromptTemplate.from_template(
    "You are a concise, truthful assistant. Answer using ONLY the context below.\\n"
    "If the context lacks the answer, say 'I don't know based on the provided context.'\\n\\n"
    "<context>\\n{context}\\n</context>\\n\\nQuestion: {question}"
)
print("Input variables expected:", rag_prompt.input_variables)
        """,
        "`ChatPromptTemplate` validates variable names automatically (`['context', 'question']`).",
    )
    nb.step(
        "Step 3: Compose the End-to-End LCEL Chain with `|`",
        "Let's snap our retriever, formatter, prompt, LLM, and string output parser together into a single runnable `rag_chain`.",
        """
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("LCEL chain ready!")
        """,
        "Calling `rag_chain.invoke('...')` executes retrieval, formatting, prompt templating, generation, and string parsing in one shot.",
    )
    nb.step(
        "Step 4: Test Both Answerable Questions and Unanswerable Refusals",
        "Let's test `rag_chain` on a specific factual question AND on a missing-fact question (Novasphere's NASDAQ stock ticker).",
        """
q_ans = "When will legacy REST API v1 be shut down completely, and what replaces it?"
q_unans = "What is Novasphere Dynamics' stock ticker symbol on NASDAQ?"
print("Answerable   :", rag_chain.invoke(q_ans))
print("Unanswerable :", rag_chain.invoke(q_unans))
        """,
        "Notice how it answers December 31, 2026 / gRPC v2 accurately for the real question, and cleanly refuses to guess a fake ticker symbol!",
    )
    nb.try_it_yourself(
        [
            ("Return both answer AND source documents", "Use `RunnableParallel` or inspect `retriever.invoke(q)` alongside `rag_chain.invoke(q)`."),
            ("Ask about Zurich pet days", "Verify the chain cites `Tuesdays and Thursdays`."),
        ],
        """
q_pets = "On which days are pets allowed in the Zurich office hub?"
used_docs = retriever.invoke(q_pets)
print("Answer :", rag_chain.invoke(q_pets))
print("Sources:", sorted({d.metadata['source'] for d in used_docs}))
        """,
    )
    nb.wrap_up(
        [
            "Forgetting `<context>` delimiters around retrieved snippets — makes it harder for LLMs to separate rules from raw document text.",
            "Omitting explicit 'say I don't know' instructions, tempting helpful models to hallucinate guesses.",
            "Returning plain answer strings in production without tracking the source filenames.",
        ],
        [
            "Grounded prompting pairs `<context>` isolation with explicit refusal rules.",
            "LCEL (`|`) chains compose retriever, formatter, prompt, LLM, and parser cleanly.",
            "Testing BOTH answerable and unanswerable queries early proves your guardrails actually work.",
        ],
        "Next up (`07_query_translation.ipynb`): What happens when users write vague or multi-part questions? Learn Multi-Query, RAG-Fusion, Decomposition, Step-Back, and HyDE!",
        [
            ("Designing Truthful Prompts", "XML delimiters + refusal rules."),
            ("LCEL Assembly Line", "How `RunnablePassthrough` and pipes work."),
            ("Source Attaching with `format_docs`", "Making citations natural and auditable."),
            ("Live Answer vs Refusal Tests", "Verifying groundedness on NASDAQ gap question."),
            ("Wrap-up", "End of Part 2 — ready for Advanced RAG!"),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    for fn in (build_nb04, build_nb05, build_nb06):
        print("Built:", fn(nb_dir).name)

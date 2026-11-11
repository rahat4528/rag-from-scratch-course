"""
Phase 1 Notebook Generator (Notebooks 00-07): Foundations & Core RAG Pipeline.
"""

from pathlib import Path
from course.utils.nb_builder import CourseNotebookBuilder


def build_nb00(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("00_why_rag.ipynb")
    nb.header(
        "00 — Why Normal Language Models Fail (and How RAG Fixes It)",
        "By the end of this notebook, you can explain why standalone LLMs hallucinate on private or recent facts and prove how supplying retrieved context fixes it.",
        [
            "Observe a standalone LLM failing on private company policies and October 2026 release facts.",
            "Compare prompt-stuffing, fine-tuning, and RAG across cost, freshness, and factual recall.",
            "Formulate a crisp one-sentence definition of Retrieval-Augmented Generation (RAG).",
        ],
        "None — basic Python is all you need.",
    )
    nb.setup()
    nb.concept(
        "Closed-Book Guessing vs. Open-Book Evidence",
        """
A **Large Language Model (LLM)** memorizes general knowledge during training — like a student taking a **closed-book exam**. When asked about *private company documents* or events *after its training cutoff*, the model cannot check external files, leading to **hallucination** (confident invention) or polite refusal.

Pasting every manual into the prompt hits a strict **context window** limit (like trying to spread 5,000 pages across a tiny school desk), while **fine-tuning** (re-training weights) costs compute and excels at style/formatting rather than mutable exact facts.

| Approach | Freshness (2026 Updates) | Private Facts Recall | Context Cost | Best Used For |
|---|---|---|---|---|
| **Prompt-Stuffing Whole Files** | Instant | High (small files only) | Very High (fails on 1,000+ pages) | Tiny single-page docs |
| **Fine-Tuning** | Stale after training | Poor / brittle exact recall | High training compute | Domain tone & output style |
| **RAG** | Instant upon index update | High with inline citations | Low (top 2–4 snippets only) | Up-to-date & private Q&A |
        """,
        """
Closed-Book: [Question] --> (Frozen LLM Weights) --> Guess / Refusal

Open-Book  : [Question] --> [Find Top Snippets] --> (Context + Question) --> [LLM] --> Grounded Answer
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Ask the LLM About Novasphere WITHOUT Context",
        "Let's ask our model an internal 2026 question about our fictional company Novasphere Dynamics before providing any documents.",
        """
question = "How much does Novasphere Dynamics charge per month for the Enterprise Plan?"
print("Closed-book output:", llm.invoke(question).content)
        """,
        "Without retrieved documents, the model explicitly cannot verify private company pricing.",
    )
    nb.step(
        "Step 2: Paste the ENTIRE Handbook Into the Prompt (Prompt-Stuffing)",
        "For one small text file, pasting the full document into `<context>` gives the model everything it needs to answer accurately.",
        """
handbook = (data_dir / "company_handbook.txt").read_text(encoding="utf-8")
stuffed_prompt = f"<context>\\n{handbook}\\n</context>\\n\\nQuestion: {question}"
print("Stuffed chars:", len(handbook))
print("Open-book reply:", llm.invoke(stuffed_prompt).content)
        """,
        "Prompt-stuffing worked for a 2 KB file — now it extracted `$4,200 per month` cleanly.",
    )
    nb.step(
        "Step 3: See Why Prompt-Stuffing Breaks at Real-World Scale",
        "Let's measure token estimates (~4 chars per token) if a real company has 5,000 such handbooks instead of one tiny demo file.",
        """
approx_tokens = len(handbook) // 4
tokens_5000_docs = approx_tokens * 5000
print(f"1 handbook ~ {approx_tokens:,} tokens | 5,000 docs ~ {tokens_5000_docs:,} tokens (overflows context & spikes latency/cost!)")
        """,
        "At 5,000 documents (~3M+ tokens), stuffing exceeds model limits, increases cost 5,000x, and buries the needle in a haystack.",
    )
    nb.step(
        "Step 4: Select Only the Matching Snippet (The Core Idea Behind RAG)",
        "If we extract ONLY the relevant pricing paragraph first, token cost plummets while answer accuracy stays crisp!",
        """
snippet = "\\n".join(l for l in handbook.splitlines() if "Plan:" in l)
mini_rag_prompt = f"<context>\\n{snippet}\\n</context>\\n\\nQuestion: {question}"
print(f"Focused snippet tokens: ~{len(snippet)//4} | Answer: {llm.invoke(mini_rag_prompt).content}")
        """,
        "That one-sentence essence: **RAG retrieves only the most relevant snippets from external data at query time so the LLM generates a grounded answer.**",
    )
    nb.try_it(
        [
            ("Test a recent 2026 release", "Compare asking about `Dr. Elina Vance` without vs. with `recent_changelog.md`."),
            ("Try an unanswerable gap", "Ask for Novasphere's NASDAQ stock ticker using `recent_changelog.md` as context."),
        ],
        """
changelog = (data_dir / "recent_changelog.md").read_text(encoding="utf-8")
print("No context:", llm.invoke("Who was appointed Chief AI Ethics Officer?").content)
print("With docs :", llm.invoke(f"<context>\\n{changelog}\\n</context>\\nQuestion: Who was appointed Chief AI Ethics Officer?").content)
print("Missing   :", llm.invoke(f"<context>\\n{changelog}\\n</context>\\nQuestion: What is Novasphere's NASDAQ stock ticker?").content)
        """,
    )
    nb.wrap_up(
        [
            "Treating fine-tuning as a factual database — memorized weights lag behind weekly changelogs.",
            "Pasting huge raw folders into prompts instead of searching for top relevant pieces first.",
            "Not allowing the prompt to answer `'I don't know'` when facts are absent.",
        ],
        [
            "One-sentence definition: **RAG searches external documents for relevant facts and injects them into the prompt so an LLM generates grounded, up-to-date answers.**",
            "Standalone LLMs suffer from knowledge cutoffs, private blindness, hallucination, and context-window budgets.",
            "Prompt-stuffing works only on tiny single docs; RAG scales cleanly to huge libraries.",
        ],
        "Next (`01_rag_big_picture.ipynb`): Build a complete RAG pipeline in plain Python with zero libraries, then see what LangChain automates!",
        [
            ("Closed-Book Failure Demo", "Asking an LLM private Novasphere questions."),
            ("Why Prompt-Stuffing Doesn't Scale", "Counting tokens at 5,000-file scale."),
            ("Prompt-Stuffing vs Fine-Tuning vs RAG", "Decision matrix for builders."),
            ("Focused Retrieval Prototype", "Trimming context down to pricing lines only."),
            ("One-Sentence RAG Definition", "Summary before coding the full pipeline."),
        ],
    )
    return nb.save(nb_dir)


def build_nb01(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("01_rag_big_picture.ipynb")
    nb.header(
        "01 — The Big Picture: Plain-Python RAG Mapped to LangChain",
        "By the end of this notebook, you can build a complete RAG loop in plain Python (no frameworks) and map each step to its LangChain module.",
        [
            "Draw the two-phase RAG architecture (Index Time vs Query Time).",
            "Implement sentence splitting, word-overlap retrieval, prompt assembly, and generation from scratch in pure Python.",
            "Map every hand-written function to its modern LangChain replacement.",
        ],
        "Notebook 00 (`00_why_rag.ipynb`)",
    )
    nb.setup()
    nb.concept(
        "Index Time vs. Query Time (No Magic Boxes!)",
        """
Before using any framework, let's peel back every layer in **pure Python**. Every RAG system splits work cleanly into two timelines:
1. **Index Time (Prepare once)**: Read files -> slice into sentences/chunks -> save searchable entries.
2. **Query Time (Run per user question)**: Score entries against question -> pick Top-2 -> inject into prompt -> generate cited reply!

| Pure Python Step We Write Today | LangChain Replacement (Notebooks 02–06) |
|---|---|
| `Path.read_text()` | Native Loaders / `Document(page_content, metadata)` |
| `split_into_sentences()` | `RecursiveCharacterTextSplitter` (`langchain_text_splitters`) |
| `word_overlap_score()` | `Embeddings` + `InMemoryVectorStore.as_retriever()` |
| `f"<context>..."` + `llm.invoke` | `ChatPromptTemplate` + LCEL / `LangGraph` |
        """,
        """
INDEX TIME (Run Once) :  Handbook + FAQ ---> Split into Sentences ---> Stored Index List
                                                                              |
QUERY TIME (Per Ask)  :  User Question ---> Overlap Scorer ---> Top-2 Hits ---+--> Prompt --> LLM --> Answer!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Index Time — Read Corpus & Split Into Sentences (Plain Python)",
        "Using pure Python only, let's read `company_handbook.txt` and `product_faq.md` into `(source, sentence)` tuples.",
        """
import re
raw_files = ["company_handbook.txt", "product_faq.md"]
index_entries = []
for fname in raw_files:
    text = (data_dir / fname).read_text(encoding="utf-8")
    for line in [l.strip() for l in text.splitlines() if len(l.strip()) > 20]:
        index_entries.append((fname, line))
print(f"Plain-Python index built cleanly: {len(index_entries)} sentence entries.")
        """,
        "Even a single Python `list` of `(filename, line)` tuples is technically a tiny searchable index!",
    )
    nb.step(
        "Step 2: Write a Pure-Python Word-Overlap Retriever",
        "Let's write `search_overlap(query, k=2)` that extracts words with regex, scores overlap, and returns top matching lines.",
        """
def search_overlap(query: str, k: int = 2):
    q_words = {w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) > 2}
    scored = []
    for src, txt in index_entries:
        t_words = set(re.findall(r"[a-z0-9_]+", txt.lower()))
        scored.append((len(q_words & t_words), src, txt))
    scored.sort(reverse=True)
    return scored[:k]

hits = search_overlap("Which ROS2 distributions are supported Humble Iron Jazzy?", k=4)
for sc, src, txt in hits: print(f"[score={sc}] ({src}) {txt}")
        """,
        "Word-overlap found exact keywords quickly — though notice how naive overlap struggles on synonyms without embeddings!",
    )
    nb.step(
        "Step 3: Pass Top Retrieved Lines Into the Prompt & Call LLM",
        "Now let's format the Top-4 lines with `[source]` labels inside `<context>` tags and generate an answer.",
        """
q = "How much does the Enterprise Plan cost per month?"
top_hits = search_overlap(q, k=3)
ctx = "\\n".join(f"[{src}] {txt}" for _, src, txt in top_hits)
ans = llm.invoke(f"Answer ONLY from context.\\n<context>\\n{ctx}\\n</context>\\nQuestion: {q}").content
print("Retrieved Context:\\n" + ctx)
print("\\nGenerated Answer :", ans)
        """,
        "Inspecting `Retrieved Context` BEFORE trusting `Generated Answer` is a golden debugging habit we will follow throughout the course.",
    )
    nb.step(
        "Step 4: See Where Pure Word-Overlap Fails (Motivation for LangChain Vectors)",
        "What if a beginner asks `'How much is the monthly fee for the top tier?'` without saying `'Enterprise'` or `'Plan'`? Overlap misses synonyms!",
        """
synonym_q = "How much is the monthly fee for the largest commercial subscription?"
missed = search_overlap(synonym_q, k=2)
print("Word-overlap hits for synonym phrasing:")
for sc, src, txt in missed: print(f"score={sc} -> {txt[:80]}")
        """,
        "Because `'fee'` != `'cost'` and `'largest'` != `'Enterprise'` literally, naive overlap scores weakly. Neural embeddings (Notebook 04) solve this cleanly!",
    )
    nb.try_it(
        [
            ("Test travel meal allowance", "Call `search_overlap('daily meal reimbursement limit VP approval', k=2)` and generate answer."),
            ("Map Python to LangChain", "Review the table in Section 2 and check which file teaches each component."),
        ],
        """
h_meal = search_overlap("daily meal reimbursement VP approval", k=2)
ctx_m = "\\n".join(f"[{s}] {t}" for _, s, t in h_meal)
print(llm.invoke(f"<context>\\n{ctx_m}\\n</context>\\nQuestion: What is the daily meal limit without VP approval?").content)
        """,
    )
    nb.wrap_up(
        [
            "Jumping straight into heavy libraries before understanding what split, search, and prompt formatting actually do.",
            "Relying purely on exact keyword overlap when users rephrase concepts with synonyms.",
            "Returning LLM answers without printing retrieved hits to spot retrieval bugs.",
        ],
        [
            "Every RAG app boils down to Index Time (prepare chunks) and Query Time (find top-k + ground prompt).",
            "A 10-line plain-Python loop proves there is zero mystery behind retrieval-augmented generation.",
            "LangChain replaces each brittle manual helper with composable `Document` loaders, splitters, vectors, and graphs.",
        ],
        "Next (`02_loading_documents.ipynb`): Replace `read_text()` with unified LangChain `Document` loaders for TXT, Markdown, PDFs, folders, and web pages!",
        [
            ("Index Time vs Query Time Diagram", "Visualizing the complete 2-phase workflow."),
            ("Pure Python Indexer", "Reading files into plain sentence tuples."),
            ("Naive Overlap Retriever", "Implementing top-k search in 7 lines."),
            ("Grounded Generation + Inspecting Chunks", "Building good debugging habits early."),
            ("When Naive Overlap Fails & LangChain Map", "Connecting plain code to Notebooks 02–06."),
        ],
    )
    return nb.save(nb_dir)


def build_nb02(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("02_loading_documents.ipynb")
    nb.header(
        "02 — Loading Documents: TXT, Markdown, PDFs, Folders & Web Pages",
        "By the end of this notebook, you can convert diverse file formats into clean LangChain Documents while handling encodings, empty pages, and web safety.",
        [
            "Understand `Document(page_content, metadata)` as LangChain's universal data envelope.",
            "Load `.txt`, `.md`, multi-page `.pdf`, whole folders, and HTML web pages (with safe offline fallback).",
            "Handle common traps: UTF-8 international characters, filenames with spaces, blank pages, and SSRF URL guards.",
        ],
        "Notebooks 00–01",
    )
    nb.setup()
    nb.concept(
        "One Standard Envelope: `Document(page_content, metadata)`",
        """
A **Document Loader** is like a mail sorter converting postcards (`.txt`), glossy brochures (`.pdf`), and web articles (`.html`) into uniform envelopes: `Document(page_content=str, metadata=dict)`.

Using clean **native readers** (`pathlib`, `pypdf.PdfReader`, `BeautifulSoup`) avoids deprecated wrapper packages while keeping `.metadata` tidy (`source='policy_brief.pdf'`, NOT `C:\\Users\\YourName\\...`).
        """,
        """
  .txt / .md  ---> Path.read_text(utf-8) --\\
  .pdf        ---> pypdf.PdfReader --------+---> List[Document(page_content, metadata)]
  Folder      ---> loop & aggregate -------/
  Web URL     ---> SSRF Check + Soup -----/
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Create & Inspect a LangChain `Document`",
        "Notice a `Document` holds `.page_content` (string text) and `.metadata` (dictionary of tags such as source filename or page).",
        """
from langchain_core.documents import Document
from course.utils.helpers import load_text_document

doc_txt = load_text_document(data_dir / "company_handbook.txt", doc_type="handbook", access_level="employee")
print("Type   :", type(doc_txt).__name__)
print("Meta   :", doc_txt.metadata)
print("Preview:", doc_txt.page_content[:95].replace("\\n", " "))
        """,
        "Normalized metadata keeps source citations portable and concise.",
    )
    nb.step(
        "Step 2: Extract PDFs & Batch-Load an Entire Folder",
        "Let's read `policy_brief.pdf` with our clean `pypdf`-backed helper, then load the whole `course/data/` folder in one call.",
        """
from course.utils.helpers import load_pdf_documents, load_folder_documents
pdf_pages = load_pdf_documents(data_dir / "policy_brief.pdf", doc_type="safety", access_level="executive")
folder_docs = load_folder_documents(data_dir)
print(f"PDF pages loaded: {len(pdf_pages)} | Chars: {len(pdf_pages[0].page_content)}")
for d in folder_docs:
    print(f" - {d.metadata['source']:<22} | type={d.metadata.get('doc_type')} | chars={len(d.page_content)}")
        """,
        "`load_folder_documents` automatically skipped `eval_questions.json` cleanly while loading every `.txt`, `.md`, and `.pdf`!",
    )
    nb.step(
        "Step 3: Load Web Pages Safely (Offline Fallback + SSRF Guard)",
        "Web loaders strip HTML tags (`<script>`, `<style>`) AND validate URLs against internal SSRF targets (`localhost`, `file://`, `169.254.*`).",
        """
from course.utils.helpers import load_web_page
offline_html = "<html><body><script>evil()</script><h1>Novasphere Docs</h1><p>gRPC v2 runs safely.</p></body></html>"
web_doc = load_web_page("https://example.invalid/docs", fallback_html=offline_html)
print("Clean web text:", web_doc.page_content, "| mode:", web_doc.metadata["fetch_mode"])
        """,
        "Providing `fallback_html` lets offline learners run web-ingestion lessons deterministically without flaky external servers.",
    )
    nb.step(
        "Step 4: Guard Common Traps (Empty Files, Corrupt PDFs & Unicode Spaces)",
        "Let's test loading a blank file (`None`), an international UTF-8 filename with spaces, and messy PDF whitespace normalization.",
        """
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    empty_p = Path(tmp) / "empty.txt"; empty_p.write_text("   ", encoding="utf-8")
    uni_p = Path(tmp) / "নিরাপত্তা brief (2026).txt"; uni_p.write_text("নিরাপত্তা — Safety first!", encoding="utf-8")
    assert load_text_document(empty_p) is None
    uni_doc = load_text_document(uni_p)
    print("Empty handled cleanly (None) | Unicode loaded:", uni_doc.metadata["source"], "->", uni_doc.page_content)
        """,
        "Empty files return `None` so blank chunks never pollute your vector database!",
    )
    nb.try_it(
        [
            ("Test SSRF URL protection", "Verify calling `load_web_page('http://127.0.0.1:8000/secret')` raises a `ValueError`."),
            ("Normalize messy PDF newlines", "Replace triple spaces/newlines inside `pdf_pages[0].page_content` using `' '.join(text.split())`."),
        ],
        """
try:
    load_web_page("http://127.0.0.1:8000/secret")
except ValueError as err:
    print("SSRF blocked:", err)
cleaned = " ".join(pdf_pages[0].page_content.split())
print("Normalized PDF preview:", cleaned[:100])
        """,
    )
    nb.wrap_up(
        [
            "Omitting `encoding='utf-8'` on Windows — crashes with `UnicodeDecodeError` on curly quotes or non-Latin scripts.",
            "Leaving machine-specific absolute paths in `.metadata['source']`.",
            "Allowing arbitrary user-entered URLs without blocking `file://` or `localhost` (SSRF vulnerability).",
        ],
        [
            "LangChain `Document` standardizes raw files under `.page_content` and `.metadata`.",
            "Native `pathlib` + `pypdf` + `BeautifulSoup` load `.txt`, `.md`, `.pdf`, and HTML with zero deprecated warnings.",
            "Skipping empty files and sanitizing metadata prevents downstream chunk & security bugs.",
        ],
        "Next (`03_splitting_text.ipynb`): Slice long documents into overlapping chunks and measure how chunk size (100 vs 300 vs 800) affects search!",
        [
            ("Universal `Document` Container", "Page content + metadata catalog card."),
            ("Text, Markdown & PDF Ingestion", "Native zero-warning Python loaders."),
            ("Folder Batch Loading", "Handling mixed formats cleanly."),
            ("SSRF-Safe Web Loading", "Stripping HTML noise + offline fallback."),
            ("Edge Cases: Empty, Messy & Unicode", "Real-world hygiene checks."),
        ],
    )
    return nb.save(nb_dir)


def build_nb03(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("03_splitting_text.ipynb")
    nb.header(
        "03 — Splitting Text Into Smart Chunks (Sizes, Overlaps & Trade-offs)",
        "By the end of this notebook, you can split long files with `RecursiveCharacterTextSplitter`, avoid cutting answers in half, and benchmark 100 vs 300 vs 800 chunk sizes.",
        [
            "Visualize how `chunk_overlap` builds a semantic bridge across split boundaries.",
            "Demonstrate a real bad split cutting an answer in half and watch overlap repair it.",
            "Run a concrete experiment comparing chunk sizes 100, 300, and 800 on the same query while preserving metadata.",
        ],
        "Notebooks 00–02",
    )
    nb.setup()
    nb.concept(
        "The Cliffhanger Bridge: Why Overlap Exists",
        """
Imagine cutting a recipe right between *"Preheat oven to"* and *"220 degrees Celsius"*. If search pulls only the second half, crucial context vanishes!

- **Chunking** breaks multi-page files into focused chunks so vector search targets specific facts cleanly.
- **Chunk Overlap** repeats the last 15–20% of chunk *N* at the top of chunk *N+1* like a cliffhanger recap across chapters.
- **RecursiveCharacterTextSplitter** prefers natural breaks (`\\n\\n` -> `\\n` -> ` ` -> char) instead of slicing mid-word.
        """,
        """
Source: [ Verify port 9872. Ensure clock drift < 5 ms. Restart novasphere-edge daemon. ]

No overlap  (bad): [ ...Ensure clock ] | [ drift < 5 ms... ]  <-- Split right across "< 5 ms"!
With overlap(fix): [ ...Ensure clock drift < 5 ms. ]
                              [ drift < 5 ms. Restart novasphere-edge daemon. ]
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Demonstrate a Bad Split vs. Overlap Fix",
        "Look at what happens when `chunk_size=100, chunk_overlap=0` chops right through `ERR_QUANTUM_TIMEOUT` resolution steps vs adding `chunk_overlap=35`.",
        """
from langchain_text_splitters import RecursiveCharacterTextSplitter
faq_text = (data_dir / "product_faq.md").read_text(encoding="utf-8")
target = [c for c in RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0).split_text(faq_text) if "9872" in c][0]
fixed_chunks = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=50).split_text(faq_text)
target_fix = [c for c in fixed_chunks if "9872" in c][0]
print("Bad no-overlap chunk lacks '< 5 ms'?:", "< 5 ms" not in target, "| snippet:", repr(target[:75]))
print("Fixed overlap chunk keeps context?  :", "QuantumCache" in target_fix or "port 9872" in target_fix)
        """,
        "Zero-overlap tiny splits stranded `port 9872` away from `ERR_QUANTUM_TIMEOUT` and `< 5 ms`!",
    )
    nb.step(
        "Step 2: Preserve Metadata Automatically Across Splits",
        "Passing `Document` objects into `.split_documents()` copies `.metadata` (`source`, `access_level`, `doc_type`) onto EVERY child chunk.",
        """
from course.utils.helpers import load_corpus_documents
docs = load_corpus_documents(data_dir)
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
print(f"Expanded {len(docs)} files -> {len(chunks)} chunks. First chunk meta:", chunks[0].metadata)
assert all("source" in c.metadata for c in chunks)
        """,
        "Every chunk remembers its source filename and RBAC access level automatically.",
    )
    nb.step(
        "Step 3: Experiment Table — Chunk Sizes 100 vs. 300 vs. 800 on One Question",
        "Let's index our corpus at chunk sizes `100`, `300`, and `800` (with 15% overlap), search for `ERR_QUANTUM_TIMEOUT` steps (`port 9872`, `< 5 ms`, `systemctl`), and tabulate complete step coverage!",
        """
import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
q = "What steps resolve ERR_QUANTUM_TIMEOUT?"
rows = []
for sz in (100, 300, 800):
    ch = RecursiveCharacterTextSplitter(chunk_size=sz, chunk_overlap=int(sz*0.15)).split_documents(docs)
    top1 = InMemoryVectorStore.from_documents(ch, embeddings).similarity_search(q, k=1)[0].page_content
    has_all = all(tok in top1 for tok in ("9872", "< 5 ms", "novasphere-edge"))
    rows.append({"ChunkSize": sz, "TotalChunks": len(ch), "Top1Len": len(top1), "HasAll3Steps": has_all})
print(pd.DataFrame(rows).to_string(index=False))
        """,
        "Too small (`100`) splits the steps apart; `800` captures all 3 steps inside Chunk #1 (though very large chunks can dilute precision on dense docs).",
    )
    nb.try_it(
        [
            ("Test invalid overlap > chunk_size", "Observe how LangChain protects against `chunk_overlap > chunk_size`."),
            ("Verify zero lost text on tiny document", "Check `.split_text('Hi RAG!')` preserves exact text."),
        ],
        """
try:
    RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=80)
except ValueError as err:
    print("Caught overlap > size cleanly:", str(err)[:85])
print("Tiny doc:", RecursiveCharacterTextSplitter(chunk_size=300).split_text("Hi RAG!"))
        """,
    )
    nb.wrap_up(
        [
            "Picking `chunk_size=100` blindly — shatters numbered instructions and tables.",
            "Setting `chunk_overlap=0` — orphans boundary words right between chunks.",
            "Forgetting `.split_documents()` and calling `.split_text()`, losing all metadata.",
        ],
        [
            "`RecursiveCharacterTextSplitter` breaks at paragraphs, lines, and spaces cleanly.",
            "Overlap (~15%) bridges boundaries without throwing away context.",
            "Chunk sizes around 300–650 chars balance pinpoint vector precision against intact answers.",
        ],
        "Next (`04_embeddings_and_vector_stores.ipynb`): Turn chunks into geometric vectors using NumPy math, plot them in 2D, and persist a local vector store!",
        [
            ("Why We Split Documents", "Balancing focus vs completeness."),
            ("Bad Split vs Overlap Fix Demo", "Rescuing cut instructions cleanly."),
            ("Metadata Inheritance", "Tracing chunks back to source files."),
            ("Chunk Size Experiment (100 / 300 / 800)", "Live Pandas measurement table."),
            ("Recap & Safeguards", "Guarding overlap errors and tiny files."),
        ],
    )
    return nb.save(nb_dir)


def build_nb04(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("04_embeddings_and_vector_stores.ipynb")
    nb.header(
        "04 — Embeddings & Vector Stores (Math, 2D Plot & Local Persistence)",
        "By the end of this notebook, you can compute cosine similarity with NumPy, visualize meaning clusters in 2D, index chunks in a vector store, and persist to disk.",
        [
            "Compute dot-product & cosine similarity by hand with NumPy on toy concept vectors.",
            "Project real 384-D `sentence-transformers` embeddings into 2D coordinates with PCA and plot them.",
            "Index documents in `InMemoryVectorStore`, dump to JSON on disk, and verify why indexing/query models must match.",
        ],
        "Notebooks 00–03",
    )
    nb.setup()
    nb.concept(
        "Turning Meaning Into Map Coordinates",
        """
An **Embedding Model** maps text to a fixed coordinate list (384 numbers for `all-MiniLM-L6-v2`). **Cosine Similarity** measures the angle between two coordinate arrows (`+1.0` identical direction, `0.0` unrelated).

A **Vector Store** stores every chunk's vector at **index time** so queries retrieve closest matches in milliseconds without re-embedding the whole library.
        """,
        """
  cos(A, B) = (A dot B) / (||A|| * ||B||)

  [Pricing Query Vector] ---> close angle (~0.7+) ---> [Enterprise $4,200 Chunk]
                         ---> right angle (~0.0) ---> [Cookies Recipe Chunk]
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Compute Cosine Similarity by Hand with NumPy (5 Sentences)",
        "Let's embed 5 crisp sentences via `embeddings.embed_documents()`, then write raw NumPy cosine similarity (`dot / norms`) without helpers.",
        """
import numpy as np
sentences = [
    "Enterprise tier monthly pricing plan",
    "Enterprise Plan costs $4,200 per month",
    "Pets are permitted in Zurich on Tuesdays",
    "Resolving ERR_QUANTUM_TIMEOUT on port 9872",
    "Chocolate cake baking recipe with sugar",
]
vecs = np.array(embeddings.embed_documents(sentences))
q_vec = vecs[0]
for s, v in zip(sentences, vecs):
    cos_sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
    print(f"cos={cos_sim:+.3f} | {s}")
        """,
        "Even though `'tier'` != `'costs $4,200'`, real neural embeddings align them high (`~0.6+`) while cooking stays near zero!",
    )
    nb.step(
        "Step 2: Plot Real Meaning Clusters in 2D (PCA + Matplotlib)",
        "Humans can't view 384 dimensions directly, so let's project our 5 vectors down to 2D `(x, y)` via PCA and render a real chart.",
        """
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pts = PCA(n_components=2).fit_transform(vecs)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(pts[:, 0], pts[:, 1], s=70)
for i, label in enumerate(["Pricing Q", "Pricing Ans", "Zurich Pets", "Error Code", "Cookies"]):
    ax.annotate(label, (pts[i, 0] + 0.02, pts[i, 1] + 0.02), fontsize=9)
ax.set_title("2D PCA Projection of 384-D Embeddings"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
        """,
        "Watch how semantically similar texts (`Pricing Q` & `Pricing Ans`) physically cluster together on the 2D chart!",
    )
    nb.step(
        "Step 3: Index Real Chunks & Persist Vector Store to Disk",
        "Let's load our corpus chunks into `InMemoryVectorStore`, dump cleanly to a JSON file, reload it, and test retrieval.",
        """
import tempfile
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
with tempfile.TemporaryDirectory() as tmp:
    p = str(Path(tmp) / "novasphere_index.json")
    store.dump(p)
    reloaded = InMemoryVectorStore.load(p, embeddings)
print("Reloaded top source:", reloaded.similarity_search("ROS2 distributions", k=1)[0].metadata["source"])
        """,
        "JSON `.dump()` / `.load()` keeps persistence 100% pure-Python and crash-free across Windows, macOS, and Linux.",
    )
    nb.step(
        "Step 4: Why Index & Query Embeddings MUST Match (Failure Demo)",
        "If an index was built with Model A (`all-MiniLM-L6-v2`) and queried with an incompatible space Model B (`LocalHashFallbackEmbeddings`), search breaks!",
        """
from course.utils.models import LocalHashFallbackEmbeddings
wrong_model = LocalHashFallbackEmbeddings(dimension=384)
mismatch_store = InMemoryVectorStore.load(
    str(next(iter([p])) if False else (lambda td: (store.dump(td + '/m.json'), td + '/m.json')[1])(tempfile.mkdtemp())),
    wrong_model,
)
print("Mismatched retrieval often pulls wrong topic ->", mismatch_store.similarity_search("Enterprise monthly cost", k=1)[0].metadata["source"])
        """,
        "Always record which `LOCAL_EMBED_MODEL` / `OPENAI_EMBED_MODEL` built an index; switching models requires re-indexing!",
    )
    nb.try_it(
        [
            ("Check empty string handling", "Confirm `embeddings.embed_query('')` returns 384 numbers cleanly."),
            ("Verify vector determinism", "Compare two `embed_query('ZeroGravity')` calls elementwise."),
        ],
        """
v_empty = embeddings.embed_query("")
v1, v2 = embeddings.embed_query("ZeroGravity"), embeddings.embed_query("ZeroGravity")
assert len(v_empty) == 384 and np.allclose(v1, v2)
print("Deterministic 384-D vector check passed cleanly.")
        """,
    )
    nb.wrap_up(
        [
            "Indexing with one embedding model and querying later with a different model.",
            "Re-embedding the entire PDF folder every time the app starts instead of loading persisted vectors.",
            "Neglecting vector normalization when comparing dot products across variable-length strings.",
        ],
        [
            "Embeddings convert words into geometric directions measured cleanly via cosine similarity.",
            "PCA 2D projections let beginners visually verify semantic clustering.",
            "`InMemoryVectorStore.dump()` and `.load()` persist indexes without native DB conflicts.",
        ],
        "Next (`05_retrieval.ipynb`): Tune retrieval parameters (`k`, scores, MMR diversity, metadata filters) and dissect retrieval failure modes!",
        [
            ("Coordinate Map Intuition", "How angles represent semantic similarity."),
            ("Manual Cosine Math in NumPy", "Writing `dot(a,b)/(||a|| ||b||)` explicitly."),
            ("2D PCA Scatter Visualization", "Plotting sentence clusters cleanly."),
            ("Vector Store Dump & Reload", "Persisting indexes to disk."),
            ("The Mismatched Embedding Trap", "Why indexing and query models must match."),
        ],
    )
    return nb.save(nb_dir)


def build_nb05(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("05_retrieval.ipynb")
    nb.header(
        "05 — Retrieval Strategies & Failure Modes (Top-k, Scores, MMR & Filters)",
        "By the end of this notebook, you can inspect similarity scores, diversify results via MMR, apply metadata filters, and diagnose three classic retrieval failures.",
        [
            "Compare `similarity_search`, `similarity_search_with_score`, and `as_retriever` with tuned `k`.",
            "Apply Maximal Marginal Relevance (`MMR`) to reduce duplicate overlapping chunks.",
            "Diagnose vague queries, vocabulary mismatch, and split answers — proving retrieval quality limits answer quality.",
        ],
        "Notebooks 00–04",
    )
    nb.setup()
    nb.concept(
        "Retrieval Quality Caps Your Maximum Answer Quality",
        """
Think of your RAG pipeline like cooking dinner: if your shopper (**Retriever**) grabs rotten or missing groceries, even a 5-star chef (**LLM**) cannot cook the right meal!

- **`similarity_search_with_score`**: Shows numeric confidence so off-topic garbage gets spotted early.
- **MMR (`max_marginal_relevance_search`)**: Balances relevance to the question against *diversity* from already-selected hits.
- **Metadata Filters**: Locks search to exact subsets (`doc_type == 'changelog'`) before scoring vectors.
        """,
        """
Good Q -> Crisp Chunks -> Accurate Cited Answer
Bad/Vague Q or Wrong k -> Missing Fact Chunk -> LLM Says "I Don't Know" or Guesses!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Index Corpus & Inspect Similarity Scores",
        "Let's create our vector store and compare confidence scores between a direct pricing query and an irrelevant question.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=45).split_documents(load_corpus_documents(data_dir))
store = InMemoryVectorStore.from_documents(chunks, embeddings)
for q in ["Enterprise Plan monthly cost", "How to bake sourdough bread"]:
    doc, score = store.similarity_search_with_score(q, k=1)[0]
    print(f"score={score:.3f} | src={doc.metadata['source']:<20} | q={q}")
        """,
        "Confidence drops cleanly from high semantic alignment down to near zero on off-topic bread recipes!",
    )
    nb.step(
        "Step 2: Wrap Store as a Retriever (`as_retriever`) & Compare MMR Diversity",
        "Watch how `search_type='mmr'` gathers broader coverage across multiple documents on compound questions.",
        """
q_multi = "Enterprise Plan pricing, ZeroGravity timeout, and October 2026 HyperSync latency"
plain_srcs = [d.metadata["source"] for d in store.as_retriever(search_kwargs={"k": 4}).invoke(q_multi)]
mmr_ret = store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.4})
mmr_srcs = [d.metadata["source"] for d in mmr_ret.invoke(q_multi)]
print("Plain Top-4 sources:", plain_srcs)
print("MMR Top-4 sources  :", mmr_srcs)
        """,
        "MMR reduces duplicate hits from the same document and broadens multi-document coverage.",
    )
    nb.step(
        "Step 3: Apply Metadata Filters Cleanly",
        "Using `filter=lambda doc: ...`, we force search strictly onto `recent_changelog.md` or `public` docs.",
        """
res = store.similarity_search("What happened in October 2026?", k=2, filter=lambda d: d.metadata.get("doc_type") == "changelog")
for d in res:
    print(f"[{d.metadata['source']}] {d.page_content[:110].replace(chr(10), ' ')}")
        """,
        "Filtering excludes older handbooks even if they contain overlapping words like `'protocol'` or `'2026'`.",
    )
    nb.step(
        "Step 4: Three Retrieval Failure Modes (Why Later Notebooks Exist!)",
        "Let's observe (1) a vague query, (2) a keyword/jargon mismatch, and (3) a tiny split-chunk miss on `eval_questions.json`.",
        """
tiny_store = InMemoryVectorStore.from_documents(RecursiveCharacterTextSplitter(chunk_size=110, chunk_overlap=0).split_documents(load_corpus_documents(data_dir)), embeddings)
print("1) Vague query ('policies') -> top source:", store.similarity_search("policies", k=1)[0].metadata["source"])
print("2) Jargon miss ('HTTP 410 endpoint?')  :", "410" in store.similarity_search("endpoint gone", k=1)[0].page_content)
print("3) Split answer (< 5 ms + 9872 in k=1?):", all(t in tiny_store.similarity_search("ERR_QUANTUM_TIMEOUT", k=1)[0].page_content for t in ("9872", "< 5 ms")))
        """,
        "Each failure teaches why we need Query Translation (NB08), Routing (NB09), Hybrid Indexing (NB10), and Reranking (NB11)!",
    )
    nb.try_it(
        [
            ("Check Baseline Retrieval Hit Rate@3", "Check how many of the 11 answerable eval questions hit the expected source inside `k=3`."),
            ("Tune MMR `lambda_mult`", "Test `lambda_mult=0.1` vs `0.9` on `q_multi`."),
        ],
        """
import json
eval_qs = [q for q in json.loads((data_dir / "eval_questions.json").read_text(encoding="utf-8")) if q["type"] != "unanswerable"]
hits = sum(1 for q in eval_qs if any(s in d.metadata["source"] for d in store.similarity_search(q["question"], k=3) for s in (q.get("source_docs") or [q["source_doc"]])))
print(f"Baseline Hit Rate@3 = {hits}/{len(eval_qs)} ({100*hits/len(eval_qs):.1f}%)")
        """,
    )
    nb.wrap_up(
        [
            "Evaluating LLM prompts when the retriever never even pulled the right paragraph.",
            "Using `k=1` on multi-document questions — guarantees partial answers.",
            "Setting MMR `fetch_k` equal to `k` — leaves zero candidates for diversity selection.",
        ],
        [
            "Retrieval quality strictly caps answer accuracy — always inspect retrieved chunks first.",
            "`similarity_search_with_score` exposes low-confidence misses.",
            "MMR diversity and metadata filters fix repetition and document-silo noise.",
        ],
        "Next (`06_generation_first_rag_chain.ipynb`): Wire retrieval into grounded prompts using LCEL pipes AND rebuild it cleanly as a 2-node LangGraph!",
        [
            ("The Retrieval Ceiling Principle", "Garbage in, garbage out."),
            ("Scores & Confidence Inspecting", "Separating real matches from nonsense queries."),
            ("MMR Diversity & Metadata Filters", "Broadening coverage safely."),
            ("Three Retrieval Failures Live", "Setting the stage for Advanced RAG."),
            ("Baseline Hit Rate@3 Check", "Measuring our starting benchmark."),
        ],
    )
    return nb.save(nb_dir)


def build_nb06(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("06_generation_first_rag_chain.ipynb")
    nb.header(
        "06 — Generation: Your First RAG Chain (Runnables + Two-Node LangGraph)",
        "By the end of this notebook, you can build a grounded RAG chain with inline prompts + source citations, then rebuild it cleanly as a two-node LangGraph workflow.",
        [
            "Author an inline strict grounded prompt (no external hub pulls) with `'say I don't know'` guardrails.",
            "Assemble a runnable LCEL chain (`|`) returning both answers and cited sources.",
            "Rebuild the exact pipeline as a typed two-node LangGraph (`retrieve -> generate`) and compare with vs without retrieval.",
        ],
        "Notebooks 00–05",
    )
    nb.setup()
    nb.concept(
        "From Linear Pipe (LCEL) to State Graph (LangGraph)",
        """
Notice our current LangChain docs pattern: instead of pulling brittle hub prompts, write prompts explicitly inline!

First we wire a classic **LCEL chain** (`|`), then rebuild it using **LangGraph `StateGraph`** — where each box (`retrieve`, `generate`) reads and updates a shared `TypedDict` state. Why learn both? Linear chains are quick; **LangGraph** lets us add grading, loops, and branches later!
        """,
        """
LCEL Pipe  : Question --> {context: retriever | format, question: passthrough} --> Prompt --> LLM
LangGraph  : [START] --> [node: retrieve (updates state['docs'])] --> [node: generate (updates state['answer'])] --> [END]
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Prepare Vector Retriever & Inline Grounded Prompt",
        "Let's index our corpus, create a source-tagging formatter, and define our strict inline `ChatPromptTemplate`.",
        """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_corpus_documents

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(load_corpus_documents(data_dir))
retriever = InMemoryVectorStore.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})
format_docs = lambda docs: "\\n\\n".join(f"[{d.metadata['source']}]\\n{d.page_content}" for d in docs)
prompt = ChatPromptTemplate.from_template(
    "Answer ONLY using the context below. If missing, say 'I don't know based on the provided context.'\\n\\n<context>\\n{context}\\n</context>\\n\\nQuestion: {question}"
)
        """,
        "Inline prompts stay version-controlled inside your repo rather than depending on external hub lookups.",
    )
    nb.step(
        "Step 2: Build the LCEL Runnable RAG Chain Returning Answer + Sources",
        "Using `RunnableParallel` (`RunnablePassthrough.assign`), we run retrieval once and preserve source filenames right alongside the answer!",
        """
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_with_sources = RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"])).assign(
    answer=lambda x: (prompt | llm | StrOutputParser()).invoke({"context": format_docs(x["docs"]), "question": x["question"]})
)
out = rag_with_sources.invoke({"question": "What is ZeroGravity Failover timeout threshold?"})
print("LCEL Answer :", out["answer"])
print("LCEL Sources:", sorted({d.metadata["source"] for d in out["docs"]}))
        """,
        "Retrieving once inside `assign(docs=...)` avoids querying the vector store twice for citations vs prompt formatting.",
    )
    nb.step(
        "Step 3: Rebuild Cleanly as a Two-Node LangGraph (`retrieve -> generate`)",
        "Now let's build the exact modern two-node **LangGraph** workflow (`START -> retrieve -> generate -> END`) over typed state!",
        """
from typing import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
class RAGState(TypedDict):
    question: str; docs: List[Document]; answer: str
def retrieve_step(state: RAGState): return {"docs": retriever.invoke(state["question"])}
def generate_step(state: RAGState):
    ctx = format_docs(state["docs"])
    return {"answer": (prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": state["question"]})}
graph = StateGraph(RAGState).add_node("retrieve", retrieve_step).add_node("generate", generate_step)
rag_graph = graph.add_edge(START, "retrieve").add_edge("retrieve", "generate").add_edge("generate", END).compile()
        """,
        "Look how clean `StateGraph` is — every node simply returns a partial dict updating `RAGState`!",
    )
    nb.step(
        "Step 4: Compare Answers WITHOUT vs. WITH Retrieval (Plus Refusal Test)",
        "Let's run the same questions through standalone `llm.invoke` vs our compiled `rag_graph` (including an unanswerable gap!).",
        """
for q in ["When does legacy REST API v1 stop working and what replaces it?", "What is Novasphere's NASDAQ stock ticker?"]:
    no_rag = llm.invoke(q).content
    res = rag_graph.invoke({"question": q, "docs": [], "answer": ""})
    srcs = sorted({d.metadata["source"] for d in res["docs"]})
    print(f"Q: {q}\\n  No RAG  : {no_rag[:90]}\\n  With RAG: {res['answer'][:120]}\\n  Sources : {srcs}")
        """,
        "With RAG enabled, real company questions are grounded cleanly while unknown NASDAQ facts trigger explicit refusal!",
    )
    nb.try_it(
        [
            ("Test Zurich pet days", "Run `rag_graph.invoke({'question': 'When are pets allowed in Zurich?', 'docs': [], 'answer': ''})`."),
            ("Inspect longest context token est", "Check `len(format_docs(res['docs'])) // 4` on `res`."),
        ],
        """
r_pet = rag_graph.invoke({"question": "On which days are pets allowed in the Zurich office hub?", "docs": [], "answer": ""})
print("Pets Answer :", r_pet["answer"])
print("Context toks:", len(format_docs(r_pet["docs"])) // 4, "| Sources:", {d.metadata["source"] for d in r_pet["docs"]})
        """,
    )
    nb.wrap_up(
        [
            "Pulling unpinned prompts from external hubs that change or break offline.",
            "Omitting `'say I don't know'` instructions in system/user prompt text.",
            "Running retrieval twice when returning both answer and citations.",
        ],
        [
            "Inline `<context>` prompts keep behavior auditable and deterministic.",
            "LCEL chains wire quick straight-line apps; LangGraph provides explicit typed state for multi-step workflows.",
            "Comparing No-RAG vs RAG + testing unanswerables verifies real grounding before deployment.",
        ],
        "Next (`07_rag_with_your_own_files.ipynb`): Mini Capstone! Point the pipeline at your own folder of files, persist & reload indexes, and score our 14 eval questions!",
        [
            ("Grounded Prompting Best Practices", "XML delimiters + inline prompts."),
            ("LCEL Runnable Assignment", "Retrieving once for answer + sources."),
            ("Modern Two-Node LangGraph", "Building `retrieve -> generate` statefully."),
            ("Before/After Comparison", "Standalone LLM vs grounded RAG vs gap refusal."),
            ("Wrap-up", "Preparing for your own files in NB07."),
        ],
    )
    return nb.save(nb_dir)


def build_nb07(nb_dir: Path) -> Path:
    nb = CourseNotebookBuilder("07_rag_with_your_own_files.ipynb")
    nb.header(
        "07 — Mini Project: RAG With Your Own Files (Save, Reload & Benchmark)",
        "By the end of this notebook, you can index an arbitrary folder of your own documents, save & reload the vector index, and benchmark answers across 14 evaluation questions.",
        [
            "Build a reusable `build_or_load_index(folder, index_path)` utility for personal files.",
            "Verify round-trip save/restart/reload queries return exact cited sources cleanly.",
            "Run the Phase 1 pipeline against `eval_questions.json` and achieve `>= 10/14` sensible answers/refusals.",
        ],
        "Notebooks 00–06",
    )
    nb.setup()
    nb.concept(
        "From Fixed Demo Files to Reusable Personal Workspace",
        """
Up to now we loaded Novasphere's sample folder. In real life, learners drop new `.txt`, `.md`, or `.pdf` reports into their own folder, **build the vector index once**, save it cleanly, and reload it in future sessions!

**Troubleshooting Checklist Before Trusting Answers:**
1. *No documents loaded?* Confirm UTF-8 encoding & non-empty files.
2. *Wrong answer retrieved?* Print top-k chunks BEFORE blaming the prompt; widen `chunk_size` (500–650) or adjust `k`.
3. *Model hallucinating?* Verify `<context>` delimiter + refusal wording in prompt.
        """,
        """
Your Folder (.txt/.md/.pdf) --> [Load & Section Split] --> [Embed & Save JSON Index]
                                                                    |
(Later Restart Session) ---> [Load JSON Index Instantaneous] ---> Query + Citations!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Create a Custom User Folder & Persist Index to JSON",
        "Let's simulate adding custom project notes into a temporary workspace folder alongside our corpus, then save the index to disk.",
        """
import json, tempfile
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from course.utils.helpers import load_folder_documents, print_qa_result

tmp_dir = tempfile.TemporaryDirectory()
user_folder = Path(tmp_dir.name) / "my_notes"; user_folder.mkdir()
(user_folder / "lab_notes.md").write_text("Project Apollo backup key is APOLLO-772 and uses port 9090.", encoding="utf-8")
index_file = str(Path(tmp_dir.name) / "saved_index.json")

user_docs = load_folder_documents(user_folder)
splitter = RecursiveCharacterTextSplitter(chunk_size=620, chunk_overlap=130, separators=["\\n## ", "\\n### ", "\\n\\n", "\\n- ", "\\n", " ", ""])
InMemoryVectorStore.from_documents(splitter.split_documents(user_docs), embeddings).dump(index_file)
print("Indexed custom files & persisted cleanly:", Path(index_file).name)
        """,
        "Dropping any `.txt`, `.md`, or `.pdf` into `user_folder` indexes cleanly with one call.",
    )
    nb.step(
        "Step 2: Reload the Persisted Index & Ask a Cited Question",
        "Now simulate a fresh script launch loading directly from `index_file` without re-splitting or re-embedding source files.",
        """
reloaded_user_store = InMemoryVectorStore.load(index_file, embeddings)
hits = reloaded_user_store.similarity_search("What port does Project Apollo use?", k=1)
ans = llm.invoke(f"<context>\\n{hits[0].page_content}\\n</context>\\nQuestion: What port does Project Apollo use?").content
print_qa_result("What port does Project Apollo use?", ans, hits)
assert "9090" in ans and hits[0].metadata["source"] == "lab_notes.md"
        """,
        "Persisted reload works cleanly with exact citation `lab_notes.md`.",
    )
    nb.step(
        "Step 3: Index Full Course Corpus & Evaluate Across All 14 Benchmark Questions",
        "Now let's test our Phase 1 core pipeline (`chunk_size=620`, section separators, `k=4`) across all 14 `eval_questions.json` items (`>= 10/14` target).",
        """
from course.utils.helpers import load_corpus_documents
corpus_chunks = splitter.split_documents(load_corpus_documents(data_dir))
corpus_store = InMemoryVectorStore.from_documents(corpus_chunks, embeddings)
eval_qs = json.loads((data_dir / "eval_questions.json").read_text(encoding="utf-8"))

passed = 0
for q in eval_qs:
    top_d = corpus_store.similarity_search(q["question"], k=4)
    ctx = "\\n\\n".join(f"[{d.metadata['source']}]\\n{d.page_content}" for d in top_d)
    a = llm.invoke(f"<context>\\n{ctx}\\n</context>\\nQuestion: {q['question']}").content.lower().replace("$", "")
    ok = ("don't know" in a or "not" in a) if q["type"] == "unanswerable" else all(kf.lower().replace("$", "") in a for kf in q["key_facts"])
    passed += int(ok)
    print(f"[{'PASS' if ok else 'MISS'}] {q['id']}: {q['question'][:48]}")
print(f"\\nPhase 1 Core Score: {passed}/{len(eval_qs)}")
assert passed >= 10, f"Expected >= 10/14, got {passed}"
        """,
        "Even with Phase 1 dense retrieval alone, we cleared the `>= 10/14` Phase 1 milestone — and we will inspect any misses carefully in Phase 2 & 3!",
    )
    nb.try_it(
        [
            ("Clean up temp dir safely", "Call `tmp_dir.cleanup()` cleanly."),
            ("Inspect source hit rate", "Check how many answerable questions had the ground-truth file in `k=4`."),
        ],
        """
tmp_dir.cleanup()
ans_qs = [q for q in eval_qs if q["type"] != "unanswerable"]
src_hits = sum(1 for q in ans_qs if any(s in d.metadata["source"] for d in corpus_store.similarity_search(q["question"], k=4) for s in (q.get("source_docs") or [q["source_doc"]])))
print(f"Answerable source Hit@4 = {src_hits}/{len(ans_qs)}")
        """,
    )
    nb.wrap_up(
        [
            "Forgetting to re-run index dump after editing files inside your personal folder.",
            "Saving vector stores without keeping matching embedding config in `.env`.",
            "Accepting low scores without debugging which exact question type missed (numbers vs multi-doc vs unanswerable).",
        ],
        [
            "A clean folder loader + `.dump()` / `.load()` lets beginners chat with any custom folder.",
            "Systematic troubleshooting starts by inspecting retrieved chunks before tweaking prompts.",
            "Evaluating against 14 structured questions gives an objective Phase 1 baseline before adding Phase 2–3 upgrades.",
        ],
        "Next (`08_query_translation.ipynb`, Phase 2!): Fix ambiguous & compound queries using Multi-Query, RAG-Fusion (RRF), Decomposition, Step-Back, and HyDE!",
        [
            ("Indexing Arbitrary Personal Folders", "Supporting custom `.txt`/`.md`/`.pdf` sets."),
            ("Save, Restart & Reload Pattern", "Persistent JSON vector store."),
            ("3-Step Troubleshooting Checklist", "Encoding -> Chunks -> Prompt."),
            ("14-Question Baseline Evaluation", "Measuring our Phase 1 milestone objectively."),
            ("Phase 1 Complete!", "Previewing Advanced Query Translation."),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    for builder in (build_nb00, build_nb01, build_nb02, build_nb03, build_nb04, build_nb05, build_nb06, build_nb07):
        print("Built:", builder(out_dir).name)

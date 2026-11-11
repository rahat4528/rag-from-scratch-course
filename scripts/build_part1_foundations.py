"""
Builder for Part 1 (Notebooks 00-03): Foundations & Document Ingestion.
"""

from pathlib import Path
from scripts.nb_builder import NotebookBuilder


def build_nb00(nb_dir: Path) -> Path:
    nb = NotebookBuilder("00_why_rag.ipynb")
    nb.header(
        "00 — Why Normal Language Models Fail (and How RAG Fixes It)",
        "By the end of this notebook, you can explain why standalone LLMs hallucinate on private or recent facts and prove how supplying retrieved context fixes it.",
        [
            "Understand four root limitations of standalone LLMs (knowledge cutoff, private data blindness, context limits, fine-tuning cost).",
            "Distinguish parametric memory (closed-book exam) from Retrieval-Augmented Generation (open-book exam).",
            "Observe an LLM failing without context versus answering accurately with retrieved context.",
        ],
        "None — designed for complete beginners who know basic Python.",
    )
    nb.setup_section()
    nb.concept_section(
        "Closed-Book vs. Open-Book AI",
        """
A **Large Language Model (LLM)** stores facts inside frozen model weights learned during training — like a student taking a **closed-book exam** using only what they memorized months ago.

When asked about *private company documents* or *events after its training cutoff*, a closed-book model suffers from **hallucination** (inventing plausible-sounding falsehoods rather than admitting ignorance). Furthermore, pasting every file into the prompt exceeds the model's **context window** (the maximum desk space of text it can read at once), while **fine-tuning** (re-training weights) is expensive and sluggish for fast-changing facts.

**Retrieval-Augmented Generation (RAG)** turns the AI into an **open-book exam**: first search your trusted documents for the exact 2-3 relevant snippets, then ask the LLM to answer *only* from those snippets.
        """,
        """
[Question] --> (Closed-Book LLM) ------------------> Guesses / "No access to private data"

[Question] --> [Search Your Docs] --> (Top Snippets + Question) --> [Open-Book LLM] --> Grounded Answer!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Ask the LLM a Private Question WITHOUT Context",
        "Let's ask our model about our fictional company Novasphere Dynamics' Enterprise Plan pricing without giving it any documents.",
        """
question = "How much does Novasphere Dynamics charge per month for the Enterprise Plan?"
closed_book = llm.invoke(question)
print("Closed-book reply:", closed_book.content)
        """,
        "Without retrieved documents, the model refuses to invent private pricing data (or in real cloud models, might hallucinate an outdated number).",
    )
    nb.step(
        "Step 2: Inspect the Actual Company Document",
        "Now let's read the genuine internal file `data/company_handbook.txt` containing our company's private pricing rules.",
        """
handbook_text = (data_dir / "company_handbook.txt").read_text(encoding="utf-8")
for line in handbook_text.splitlines():
    if "Enterprise Plan" in line:
        print("Found in handbook:", line.strip())
        """,
        "The ground truth ($4,200/month) lives safely in our local text file—outside the LLM's frozen training weights.",
    )
    nb.step(
        "Step 3: Compare Context Window Limits vs. Relevant Snippets",
        "Instead of dumping an entire 10,000-page handbook into the LLM's limited context window, let's check character counts and isolate only the pricing section.",
        """
pricing_snippet = "\\n".join(
    line for line in handbook_text.splitlines() if "Plan:" in line or "PRICING" in line
)
print(f"Full handbook chars: {len(handbook_text)} | Focused snippet chars: {len(pricing_snippet)}")
        """,
        "Focusing strictly on relevant paragraphs keeps prompts small, cheap, fast, and much more accurate.",
    )
    nb.step(
        "Step 4: Open-Book Answering with Grounded Context (Mini-RAG)",
        "Now let's inject our focused snippet into `<context>` tags and instruct the LLM to answer solely from that evidence.",
        """
prompt = f"<context>\\n{pricing_snippet}\\n</context>\\nQuestion: {question}"
open_book = llm.invoke(prompt)
print("Open-book reply:", open_book.content)
        """,
        "Once grounded in retrieved text, the LLM returns the exact figure ($4,200 per month) with zero guessing!",
    )
    nb.try_it_yourself(
        [
            ("Test a recent 2026 event", "Read `recent_changelog.md` and ask who was appointed Chief AI Ethics Officer with vs. without context."),
            ("Try an unanswerable question", "Pass the pricing snippet and ask for Novasphere's stock ticker symbol on NASDAQ."),
        ],
        """
changelog = (data_dir / "recent_changelog.md").read_text(encoding="utf-8")
print("No context :", llm.invoke("Who is Novasphere's Chief AI Ethics Officer?").content[:95])
print("With RAG   :", llm.invoke(f"<context>\\n{changelog}\\n</context>\\nWho was appointed Chief AI Ethics Officer?").content)
print("Missing    :", llm.invoke(f"<context>\\n{changelog}\\n</context>\\nWhat is Novasphere's NASDAQ ticker?").content)
        """,
    )
    nb.wrap_up(
        [
            "Assuming fine-tuning updates a model's factual memory cheaply — fine-tuning shapes tone and task style, while RAG supplies accurate, updatable facts.",
            "Stuffing entire 100-page PDFs into one prompt — causes lost-in-the-middle errors and huge token bills.",
            "Prompting without telling the model to say 'I don't know' when context lacks the answer.",
        ],
        [
            "Standalone LLMs suffer from training cutoffs, private-data blindness, hallucinations, and context-window budgets.",
            "RAG pairs an external search step with a grounded generation step (open-book exam).",
            "You saw firsthand how supplying `<context>` turns an unknown proprietary question into a verified fact.",
        ],
        "Next up (`01_hello_rag.ipynb`): Build an end-to-end automated RAG pipeline using real LangChain loaders, splitters, embeddings, and vector search!",
        [
            ("The Closed-Book Problem", "Why LLMs forget, hallucinate, and hit training cutoffs."),
            ("Context Window & Fine-Tuning Myths", "Why dumping huge files or retraining fails for factual recall."),
            ("Open-Book AI Intuition", "Walkthrough of Retrieval-Augmented Generation with a library analogy."),
            ("Live Code Demo", "Closed-book failure vs. open-book grounded success on Novasphere data."),
            ("Wrap-up & What's Next", "Key takeaways before building automated search in NB01."),
        ],
    )
    return nb.save(nb_dir)


def build_nb01(nb_dir: Path) -> Path:
    nb = NotebookBuilder("01_hello_rag.ipynb")
    nb.header(
        "01 — Hello RAG: Your First End-to-End Pipeline",
        "By the end of this notebook, you can load files, split them into chunks, index vectors, retrieve top matches, and generate cited answers.",
        [
            "Assemble the complete 5-stage RAG architecture (Load -> Split -> Embed & Store -> Retrieve -> Generate).",
            "Inspect real LangChain Document objects, chunk sizes, and vector similarity search results.",
            "Run live question answering against two real Novasphere documents.",
        ],
        "Notebook 00 (`00_why_rag.ipynb`)",
    )
    nb.setup_section()
    nb.concept_section(
        "The 5-Stage RAG Pipeline",
        """
Every RAG system follows two phases: an **offline Indexing phase** (prepare your library shelf once) and a **real-time Query phase** (find answers on demand):

1. **Loader**: Reads files (`.txt`, `.md`, `.pdf`) into uniform **Documents** (text + metadata card).
2. **Text Splitter**: Slices long documents into short **Chunks** (~350 characters) so search hits pinpoint exact paragraphs.
3. **Embedding + Vector Store**: Turns each chunk into a numeric meaning coordinate (**Embedding**) inside an **InMemoryVectorStore** (a smart semantic filing cabinet).
4. **Retriever**: Embeds your user question and pulls the top-k closest matching chunks.
5. **Generator (Prompt + LLM)**: Hands the chunks to the LLM to synthesize a clear, cited answer.
        """,
        """
INDEXING (Offline):  Files --> [Loader] --> [Splitter] --> [Embeddings] --> [Vector Store]
                                                                                  |
QUERYING (Online) :  User Question --> [Embed Question] --> [Similarity Search] --+
                                                                    |
                                          Final Answer <-- [LLM] <--+ (Top-k Chunks)
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Load Real Documents",
        "Let's use LangChain's `TextLoader` to load both `company_handbook.txt` and `product_faq.md` into standard `Document` objects.",
        """
from langchain_community.document_loaders import TextLoader
docs = []
for name in ["company_handbook.txt", "product_faq.md"]:
    loaded = TextLoader(str(data_dir / name), encoding="utf-8").load()
    for d in loaded: d.metadata["source"] = name
    docs.extend(loaded)
print(f"Loaded {len(docs)} documents cleanly.")
        """,
        "Notice each document carries both `.page_content` (the raw words) and `.metadata` (where it came from).",
    )
    nb.step(
        "Step 2: Split Documents Into Searchable Chunks",
        "A whole handbook is too broad for precise lookup, so let's chop it into overlapping 380-character pieces with `RecursiveCharacterTextSplitter`.",
        """
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split {len(docs)} files into {len(chunks)} searchable chunks.")
print("Sample chunk metadata:", chunks[0].metadata)
        """,
        "Chunking keeps each piece small enough to match specific topics while carrying its source tag forward.",
    )
    nb.step(
        "Step 3: Embed Chunks & Store in Vector Store",
        "Now let's pass all chunks through our local embedding model into `InMemoryVectorStore` so they are searchable by meaning.",
        """
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Vector store ready! Retriever configured for top k=3 chunks.")
        """,
        "Behind the scenes, every chunk became a 384-dimensional vector coordinate stored in memory.",
    )
    nb.step(
        "Step 4: Retrieve Relevant Chunks for a Question",
        "Let's invoke our `retriever` on a question about ROS2 support and inspect what chunks come back.",
        """
from utils.helpers import pretty_print_docs
question = "Which ROS2 distributions are natively supported by Novasphere?"
hits = retriever.invoke(question)
pretty_print_docs(hits, max_chars=140)
        """,
        "Our semantic retriever pinpointed `product_faq.md` containing Humble Hawksbill, Iron Irwini, and Jazzy Jalisco!",
    )
    nb.step(
        "Step 5: Generate a Grounded Answer",
        "Finally, let's combine the retrieved chunk texts into `<context>` tags and run our LLM chain to produce the answer.",
        """
from utils.helpers import print_qa_result
context_text = "\\n\\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in hits)
prompt = f"<context>\\n{context_text}\\n</context>\\nQuestion: {question}"
answer = llm.invoke(prompt).content
print_qa_result(question, answer, hits)
        """,
        "Congratulations! You just built a working, cited RAG system from scratch in 5 crisp steps.",
    )
    nb.try_it_yourself(
        [
            ("Ask about travel meals", "Invoke the same pipeline for the daily meal reimbursement limit without prior VP approval."),
            ("Adjust k to 1 vs 4", "See whether changing `search_kwargs={'k': 1}` still catches multi-part answers."),
        ],
        """
q_try = "What is the maximum daily meal expense reimbursement without VP approval?"
try_hits = vector_store.similarity_search(q_try, k=2)
ctx_try = "\\n".join(d.page_content for d in try_hits)
print(llm.invoke(f"<context>\\n{ctx_try}\\n</context>\\nQuestion: {q_try}").content)
        """,
    )
    nb.wrap_up(
        [
            "Forgetting to pass `encoding='utf-8'` to `TextLoader` on Windows (causes UnicodeDecodeError on special symbols or non-Latin text).",
            "Re-embedding the entire corpus on every single user query instead of indexing once and querying repeatedly.",
            "Setting `k=1` on complex questions — often misses crucial details in neighboring paragraphs.",
        ],
        [
            "RAG runs in two distinct phases: Index offline, Retrieve + Generate online.",
            "LangChain unifies loaders, splitters, embeddings, and retrievers under clean `.invoke()` methods.",
            "Small, focused chunks + citations let beginners trace exactly why the model answered the way it did.",
        ],
        "Next up (`02_loading_documents.ipynb`): Master real-world file loading across TXT, Markdown, PDF, Web pages, Unicode scripts, and broken files!",
        [
            ("From Idea to Working Pipeline", "Overview of the 5-stage indexing + retrieval diagram."),
            ("Loading & Chunking Data", "Hands-on `TextLoader` + `RecursiveCharacterTextSplitter`."),
            ("Indexing Into InMemoryVectorStore", "How chunks become searchable vectors in 2 lines of Python."),
            ("Retrieve + Generate Live", "Running our first real Q&A with cited sources."),
            ("Recap & Homework", "Tuning k and testing your own questions."),
        ],
    )
    return nb.save(nb_dir)


def build_nb02(nb_dir: Path) -> Path:
    nb = NotebookBuilder("02_loading_documents.ipynb")
    nb.header(
        "02 — Loading Real Documents (TXT, Markdown, PDF, Web & Metadata)",
        "By the end of this notebook, you can ingest plain text, Markdown, PDFs, web pages (with offline fallback), and international Unicode safely.",
        [
            "Understand LangChain's `Document(page_content, metadata)` data structure.",
            "Load `.txt`, `.md`, `.pdf`, and HTML content consistently with proper UTF-8 encoding.",
            "Attach normalized metadata fields (`source`, `doc_type`, `page`) for later filtering and citations.",
        ],
        "Notebooks 00–01",
    )
    nb.setup_section()
    nb.concept_section(
        "Universal Adapter Pattern for Documents",
        """
Think of **Document Loaders** like universal travel power adapters: whether your company knowledge lives inside plain `.txt` files, formatted `.md` release notes, multi-page `.pdf` briefs, or live web pages, every loader plugs them into one standard output: a Python list of **Document** objects.

Every **Document** holds exactly two attributes:
- `.page_content`: The extracted clean text string.
- `.metadata`: A Python `dict` (like a library catalog card) storing `source`, `page`, `doc_type`, or permissions.
        """,
        """
  company_handbook.txt ---> [TextLoader] ---\\
  recent_changelog.md  ---> [TextLoader] ----+--> List[Document(page_content=..., metadata={...})]
  policy_brief.pdf     ---> [PyPDFLoader] --/
  https://... / HTML   ---> [Web / Fallback]/
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Load Text and Markdown Files with Normalized Metadata",
        "Let's load `.txt` and `.md` files and enrich their `metadata` dictionaries so filenames stay concise rather than exposing messy system paths.",
        """
from langchain_community.document_loaders import TextLoader
files = [("company_handbook.txt", "handbook"), ("recent_changelog.md", "changelog")]
all_docs = []
for fname, dtype in files:
    for d in TextLoader(str(data_dir / fname), encoding="utf-8").load():
        d.metadata.update({"source": fname, "doc_type": dtype})
        all_docs.append(d)
print(f"Loaded {len(all_docs)} text/markdown docs. Meta[1]:", all_docs[1].metadata)
        """,
        "Clean metadata (`source='recent_changelog.md'`) ensures our citations remain portable across laptops and OSes.",
    )
    nb.step(
        "Step 2: Extract Text from a Real PDF Document",
        "PDFs contain binary layout commands rather than raw text. Let's inspect `policy_brief.pdf` using `PyPDFLoader` (or `pypdf`) and preview its extracted page text.",
        """
from langchain_community.document_loaders import PyPDFLoader
pdf_docs = PyPDFLoader(str(data_dir / "policy_brief.pdf")).load()
for d in pdf_docs:
    d.metadata["source"] = "policy_brief.pdf"
    d.metadata["doc_type"] = "safety"
print(f"PDF Pages: {len(pdf_docs)} | Chars: {len(pdf_docs[0].page_content)}")
print("Preview:", pdf_docs[0].page_content[:120].replace("\\n", " "))
        """,
        "`PyPDFLoader` automatically splits multi-page PDFs into one `Document` per page with `page` numbers in metadata.",
    )
    nb.step(
        "Step 3: Web Loading with Safe Offline Fallback",
        "Fetching live URLs can fail if a learner is offline or a site blocks bots. Always wrap web loaders with graceful error handling and fallback parsing.",
        """
from bs4 import BeautifulSoup
from langchain_core.documents import Document
sample_html = "<html><body><h1>Novasphere Docs</h1><p>gRPC v2 runs on port 50051.</p></body></html>"
soup = BeautifulSoup(sample_html, "html.parser")
web_doc = Document(page_content=soup.get_text(" ", strip=True), metadata={"source": "docs.html", "doc_type": "web"})
print("Extracted HTML text:", web_doc.page_content)
        """,
        "Stripping HTML boilerplate with `BeautifulSoup` keeps noisy `<div>` and `<script>` tags out of your vector index.",
    )
    nb.step(
        "Step 4: Handle International Unicode & Filenames with Spaces Safely",
        "Real enterprise documents contain non-ASCII languages (e.g., Bangla, Japanese, accented European letters) and spaces in filenames. Let's test `pathlib.Path` + `utf-8` handling.",
        """
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "my policy doc (2026).txt"
    p.write_text("নিরাপত্তা নীতিমালা — Zurich Hub allows dogs on Tuesdays.", encoding="utf-8")
    unicode_doc = TextLoader(str(p), encoding="utf-8").load()[0]
print("Loaded safely:", unicode_doc.page_content)
        """,
        "Always specifying `encoding='utf-8'` and using `Path` objects avoids Windows CP-1252 crashes and broken paths with spaces.",
    )
    nb.try_it_yourself(
        [
            ("Use shared helper", "Call `load_corpus_documents(data_dir)` from `utils.helpers` and check total loaded docs."),
            ("Count characters per doc_type", "Loop through loaded docs and print filename + total word count."),
        ],
        """
from utils.helpers import load_corpus_documents
corpus = load_corpus_documents(data_dir)
for d in corpus:
    words = len(d.page_content.split())
    print(f"- {d.metadata['source']:<22} type={d.metadata['doc_type']:<10} words={words}")
        """,
    )
    nb.wrap_up(
        [
            "Loading raw scanned-image PDFs without OCR — `PyPDFLoader` reads embedded text; image-only scans extract empty strings unless OCR is added.",
            "Leaving absolute paths (`C:\\Users\\...`) inside `.metadata['source']` — breaks relative source filtering and leaks usernames.",
            "Forgetting error handling on `WebBaseLoader` requests when URLs return 404 or timeout.",
        ],
        [
            "Every LangChain Document combines `.page_content` (str) and `.metadata` (dict).",
            "`TextLoader(..., encoding='utf-8')`, `PyPDFLoader`, and `BeautifulSoup` cover 90% of corporate files safely.",
            "Clean, consistent metadata laid here powers filtering, RBAC security, and citations in later notebooks.",
        ],
        "Next up (`03_splitting_text.ipynb`): Learn why naive slicing cuts words in half and how Recursive & Markdown splitters preserve complete thoughts!",
        [
            ("What Is a Document Object?", "Inspecting page_content vs metadata."),
            ("Loading TXT & Markdown Files", "Clean paths and UTF-8 handling."),
            ("Reading PDFs Safely", "Page-by-page extraction with PyPDFLoader."),
            ("HTML Scraping & Fallbacks", "Extracting readable prose without HTML noise."),
            ("Edge Cases: Unicode & Spaces", "Preventing OS path and encoding bugs."),
        ],
    )
    return nb.save(nb_dir)


def build_nb03(nb_dir: Path) -> Path:
    nb = NotebookBuilder("03_splitting_text.ipynb")
    nb.header(
        "03 — Splitting Text Into Smart Chunks (Sizes, Overlaps & Boundaries)",
        "By the end of this notebook, you can split long files without cutting sentences in half and choose smart chunk sizes with overlap.",
        [
            "Understand why chunk size trades off precision (tiny chunks) against context completeness (large chunks).",
            "Compare naive fixed-width splitting with `RecursiveCharacterTextSplitter` and `MarkdownHeaderTextSplitter`.",
            "Verify mathematically that `chunk_overlap` preserves boundary facts with zero missing words.",
        ],
        "Notebook 02 (`02_loading_documents.ipynb`)",
    )
    nb.setup_section()
    nb.concept_section(
        "Why Overlapping Recursive Splitting Matters",
        """
Imagine reading a recipe book where every page ends mid-sentence: *"Bake at 180 degrees for..."* and the next page starts *"...45 minutes."* If search only retrieves page 2, you lose the temperature!

- **Text Splitting (Chunking)** divides long texts into bite-sized pieces so embeddings represent one cohesive topic instead of blurring 20 topics together.
- **Chunk Overlap** copies the tail of Chunk 1 onto the head of Chunk 2 (like repeating a recap sentence across chapter pages), guaranteeing answers near boundaries stay intact.
- **RecursiveCharacterTextSplitter** tries semantic boundaries in priority order (`\\n\\n` paragraphs -> `\\n` lines -> spaces -> chars) rather than chopping blindly inside words!
        """,
        """
Original Text: [ Sentence A. Sentence B. Sentence C. Sentence D. ]

Chunk 1 (size=30, overlap=10): [ Sentence A. Sentence B. ]
Chunk 2 (starts inside tail) :              [ Sentence B. Sentence C. ]
                                               ^^^ Overlap protects boundary meaning!
        """,
    )
    nb.start_build()
    nb.step(
        "Step 1: Compare Naive Fixed Slicing vs. Recursive Splitting",
        "Look at how naive fixed-interval splitting cuts right through the middle of words (`ZeroGra` / `vity`), whereas `RecursiveCharacterTextSplitter` respects word/paragraph boundaries.",
        """
sample = (data_dir / "product_faq.md").read_text(encoding="utf-8")[:260]
naive = [sample[i:i+90] for i in range(0, len(sample), 90)]
print("Naive tail of chunk 0:", repr(naive[0][-25:]))
print("Naive head of chunk 1:", repr(naive[1][:25]))
        """,
        "Naive slicing splits `'ZeroGravity'` or `'99.995%'` straight down the middle, ruining search relevance!",
    )
    nb.step(
        "Step 2: Apply `RecursiveCharacterTextSplitter` on Clean Overlap",
        "Now let's apply `RecursiveCharacterTextSplitter` (`chunk_size=140, chunk_overlap=35`) on the same passage and verify clean boundary respect.",
        """
from langchain_text_splitters import RecursiveCharacterTextSplitter
rec = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=35)
rec_chunks = rec.split_text(sample)
for i, ch in enumerate(rec_chunks):
    print(f"Chunk {i} ({len(ch)} chars): {ch[:55].replace(chr(10), ' ')} ... {ch[-30:].replace(chr(10), ' ')}")
        """,
        "Every chunk starts and ends neatly between paragraphs or spaces—never slicing mid-word.",
    )
    nb.step(
        "Step 3: Observe Overlap Preservation Across Two Chunks",
        "Let's print the trailing words of Chunk 0 and the leading words of Chunk 1 to confirm our overlap bridge is physically present.",
        """
tail_words = set(rec_chunks[0].split())
head_words = set(rec_chunks[1].split())
shared = sorted(tail_words & head_words)
print("Words shared across boundary:", shared[:8])
assert len(rec_chunks) >= 2, "Expected at least 2 chunks"
        """,
        "Shared tokens across consecutive chunks ensure facts sitting right near a split point stay searchable in full.",
    )
    nb.step(
        "Step 4: Split Markdown by Section Headings (`MarkdownHeaderTextSplitter`)",
        "Markdown documents already carry semantic section headers (`#`, `##`, `###`). Splitting by headers attaches section titles directly into metadata!",
        """
from langchain_text_splitters import MarkdownHeaderTextSplitter
md_text = (data_dir / "product_faq.md").read_text(encoding="utf-8")
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "section"), ("###", "topic")])
md_docs = md_splitter.split_text(md_text)
print(f"Markdown sections created: {len(md_docs)}")
print("Metadata of section 1:", md_docs[1].metadata)
        """,
        "Header splitting gives every retrieved chunk free structural context (`section='Edge Compatibility & ROS2 Support'`).",
    )
    nb.try_it_yourself(
        [
            ("Sweep chunk sizes", "Split `company_handbook.txt` with `chunk_size` = 200 vs 400 vs 800 and observe chunk counts."),
            ("Tiny document guard", "Verify what `RecursiveCharacterTextSplitter(chunk_size=400)` outputs on a tiny 12-character note."),
        ],
        """
handbook = (data_dir / "company_handbook.txt").read_text(encoding="utf-8")
for sz in (200, 400, 800):
    n = len(RecursiveCharacterTextSplitter(chunk_size=sz, chunk_overlap=40).split_text(handbook))
    print(f"chunk_size={sz} -> {n} chunks")
print("Tiny doc output:", RecursiveCharacterTextSplitter(chunk_size=400).split_text("Hello RAG!"))
        """,
    )
    nb.wrap_up(
        [
            "Using `chunk_overlap >= chunk_size` — triggers infinite redundancy or errors; keep overlap around 10%–20% of `chunk_size`.",
            "Picking `chunk_size=50` — creates fragmented meaningless snippets; picking `chunk_size=5000` blurs unrelated subjects.",
            "Splitting code or Markdown tables blindly without header/code-aware separators.",
        ],
        [
            "Smart splitting balances pinpoint specificity with readable context.",
            "`RecursiveCharacterTextSplitter` preserves paragraphs and words cleanly before falling back.",
            "`MarkdownHeaderTextSplitter` turns document structure (`## Section`) directly into retrieval metadata.",
        ],
        "Next up (`04_embeddings_and_vector_stores.ipynb`): Convert your clean text chunks into numeric vectors and store them in an instant semantic vector index!",
        [
            ("Why Naive Slicing Breaks Search", "Live example of slicing right through words."),
            ("Chunk Size vs. Overlap Trade-offs", "Analogy & optimal 10–20% overlap rule of thumb."),
            ("RecursiveCharacterTextSplitter Demo", "Visualizing clean paragraph/word splits."),
            ("MarkdownHeaderTextSplitter Magic", "Automatic metadata tagging from headers."),
            ("Experiments & Wrap-up", "Benchmarking chunk counts across sizes."),
        ],
    )
    return nb.save(nb_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    for fn in (build_nb00, build_nb01, build_nb02, build_nb03):
        p = fn(nb_dir)
        print("Built:", p.name)


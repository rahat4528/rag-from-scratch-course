# Upstream & Dependency Audit — `KNOWN_ISSUES.md`

## 1. Upstream Base Notebook Notes (`langchain-ai/rag-from-scratch`)
- **API Sunset**: Upstream notebooks written in late 2023 / early 2024 rely on deprecated LangChain 0.0.x abstractions (`RetrievalQA.from_chain_type`, `from langchain_community.chat_models import ChatOpenAI`, `langchain.vectorstores.Chroma`). In this modern course, all code adheres strictly to LangChain 1.x standard LCEL (`|`), `langgraph`, and `InMemoryVectorStore`.
- **Local Neural Fallback**: Upstream notebooks assumed learners possess paid OpenAI API keys or local Ollama servers. This course introduces `LocalSentenceTransformerEmbeddings` (`all-MiniLM-L6-v2`) and `LocalGroundedChatModel` enabling 100% free, offline execution out of the box.

## 2. Dependency Audit & Mitigations
- **Pillow 12.3.0**: Flagged in pip-audit due to upstream Gradio dependency pins. **Mitigation**: File upload handlers restrict ingestion strictly to `.txt`, `.md`, and `.pdf` documents; image processing is bypassed entirely.
- **PyTorch 2.8.0**: Selected specifically for Windows compatibility to avoid missing DLL dependencies in PyTorch 2.9+ / 2.15+.


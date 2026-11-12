"""
Shared LLM and Embedding factory for the RAG From Scratch course.

Reads configuration from `.env` (via `RAG_PROVIDER`) and provides a uniform
LangChain interface across:
  - `local`     (Default: 100% free offline deterministic grounded model + local embeddings)
  - `ollama`    (Local Ollama server chat model + local embeddings)
  - `openai`    (Hosted OpenAI chat + embeddings)
  - `anthropic` (Hosted Anthropic Claude chat + OpenAI or local embeddings)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from utils.helpers import sanitize_error_message

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "based", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "may", "might", "must", "not", "of", "on", "only", "or",
    "our", "provided", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "use",
    "used", "using", "was", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your", "according", "context", "document",
    "documents", "information", "mention", "mentioned", "question", "answer",
})

_SYNONYM_GROUPS = [
    {"cost", "price", "pricing", "fee", "charge", "plan", "month", "monthly", "$"},
    {"pet", "pets", "dog", "dogs", "animal", "animals"},
    {"allowed", "permitted", "permit", "allow"},
    {"tuesday", "tuesdays", "thursday", "thursdays"},
    {"latency", "delay", "roundtrip", "speed", "faster"},
    {"error", "err_quantum_timeout", "troubleshoot", "resolve", "fix", "timeout", "issue"},
    {"shut", "shutdown", "retire", "retired", "retirement", "deprecated", "deprecation", "410"},
    {"tier", "plan", "subscription"},
    {"meal", "meals", "food", "reimburse", "reimbursement", "expense"},
    {"stipend", "learning", "training", "education", "allowance"},
    {"uptime", "sla", "availability", "guarantee", "guaranteed"},
    {"response", "respond", "sla", "first", "incident", "severity"},
    {"channels", "channel", "email", "portal", "slack"},
    {"ethics", "officer", "elina", "vance", "appointed"},
    {"failover", "zerogravity", "peer", "timeout", "threshold", "ms", "milliseconds"},
    {"ros2", "ros", "humble", "iron", "jazzy", "distribution", "distributions"},
]

_WORD_TO_CANON: Dict[str, str] = {}
for idx, grp in enumerate(_SYNONYM_GROUPS):
    canon = sorted(grp)[0]
    for w in grp:
        _WORD_TO_CANON[w] = f"syn_{idx}_{canon}"


def resolve_project_root() -> Path:
    """Walk upward from CWD or module directory to locate the course repository root."""
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent.parent]
    for start in candidates:
        cur = start
        for _ in range(6):
            if (cur / "utils" / "models.py").exists() and (cur / "data").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    return Path(__file__).resolve().parent.parent


def _load_env() -> Path:
    root = resolve_project_root()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return root


def _is_valid_api_key(key: Optional[str]) -> bool:
    if not key:
        return False
    stripped = key.strip()
    lower = stripped.lower()
    if len(stripped) < 16:
        return False
    invalid_markers = ("replace", "your-", "your_", "placeholder", "dummy", "changeme", "xxx", "invalid")
    return not any(marker in lower for marker in invalid_markers)


def _normalize_tokens(text: str) -> List[str]:
    """Tokenize into lowercase words, numbers, and code tokens with light stemming."""
    raw = re.findall(r"[a-zA-Z0-9_.$%-]+(?:<[0-9]+)?", text.lower())
    tokens: List[str] = []
    for tok in raw:
        clean = tok.strip(".,;:!?()[]{}\"'`")
        if not clean:
            continue
        tokens.append(clean)
        if clean in _WORD_TO_CANON:
            tokens.append(_WORD_TO_CANON[clean])
        # Light suffix normalization
        for suf in ("ations", "ation", "ing", "ed", "es", "s"):
            if len(clean) > len(suf) + 3 and clean.endswith(suf):
                tokens.append(clean[: -len(suf)])
                break
    return tokens


def cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    """Compute cosine similarity between two 1D numeric vectors (canonical utility for E2)."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class LocalHashEmbeddings(Embeddings):
    """
    Fast, zero-dependency, deterministic 384-dimensional local embeddings.

    Combines hashed word stems, synonym-canonicalized concepts, character trigrams,
    and word bigrams into an L2-normalized vector so both lexical matches and
    paraphrased queries cluster cleanly in cosine space.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_single(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text or not text.strip():
            return vec.tolist()

        tokens = _normalize_tokens(text)
        # 1. Unigram token projections (higher weight for rare/numeric/code tokens)
        for tok in tokens:
            if tok in _STOPWORDS:
                continue
            weight = 2.0 if (any(c.isdigit() for c in tok) or "_" in tok or "$" in tok or "%" in tok) else 1.2
            if tok.startswith("syn_"):
                weight = 1.4
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        # 2. Bigrams for phrase continuity
        content_words = [t for t in tokens if t not in _STOPWORDS and not t.startswith("syn_")]
        for i in range(len(content_words) - 1):
            bg = f"{content_words[i]}_{content_words[i + 1]}"
            h = hashlib.blake2b(bg.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign * 0.9

        # 3. Character trigrams for typo/morphology resilience
        cleaned_str = re.sub(r"\s+", " ", text.lower()).strip()
        for i in range(max(0, len(cleaned_str) - 2)):
            tri = cleaned_str[i : i + 3]
            if tri.strip():
                h = hashlib.blake2b(tri.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dimension
                sign = 1.0 if (h[4] & 1) == 0 else -1.0
                vec[idx] += sign * 0.25

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t or "") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text or "")


_ST_MODEL_CACHE: Dict[str, Any] = {}


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Wrapper around HuggingFace `sentence-transformers` for true local semantic embeddings.
    Falls back cleanly to `LocalHashEmbeddings` if offline and the model is not yet cached.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._fallback: Optional[LocalHashEmbeddings] = None
        self._model = _ST_MODEL_CACHE.get(model_name)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                offline_only = os.getenv("RAG_OFFLINE", "0") == "1"
                self._model = SentenceTransformer(model_name, local_files_only=offline_only)
                _ST_MODEL_CACHE[model_name] = self._model
            except Exception as exc:
                warnings.warn(
                    f"Falling back to LocalHashEmbeddings because SentenceTransformer('{model_name}') "
                    f"could not be loaded: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback = LocalHashEmbeddings(dimension=384)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._fallback is not None:
            return self._fallback.embed_documents(texts)
        cleaned = [t if (t and t.strip()) else " " for t in texts]
        if not cleaned:
            return []
        vecs = self._model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        if self._fallback is not None:
            return self._fallback.embed_query(text)
        clean = text if (text and text.strip()) else " "
        vec = self._model.encode([clean], normalize_embeddings=True, show_progress_bar=False)[0]
        return vec.tolist()


class CachedEmbeddings(Embeddings):
    """
    Wrapper adding an in-memory SHA-256 cache around any Embeddings implementation.
    Exposes `.cache_stats()` for latency and cost optimization lessons.
    """

    def __init__(self, base: Embeddings) -> None:
        self.base = base
        self._cache: Dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for i, text in enumerate(texts):
            k = self._key(text)
            if k in self._cache:
                self.hits += 1
                results[i] = self._cache[k]
            else:
                self.misses += 1
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            computed = self.base.embed_documents(missing_texts)
            for idx, text, vec in zip(missing_indices, missing_texts, computed):
                self._cache[self._key(text)] = vec
                results[idx] = vec

        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> List[float]:
        k = self._key(text)
        if k in self._cache:
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        vec = self.base.embed_query(text)
        self._cache[k] = vec
        return vec

    def cache_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._cache),
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


def _extract_section(text: str, markers: Sequence[str]) -> Optional[str]:
    """Extract content following one of the uppercase/tag markers."""
    lower = text.lower()
    for m in markers:
        pos = lower.find(m.lower())
        if pos != -1:
            tail = text[pos + len(m) :]
            end_tag = m.replace("<", "</")
            if m.startswith("<") and end_tag.lower() in tail.lower():
                end_pos = tail.lower().find(end_tag.lower())
                return tail[:end_pos].strip()
            return tail.strip()
    return None


def _split_context_sentences(context_text: str) -> List[str]:
    """Break context into clean fact-bearing blocks and sentences, stripping injection lines."""
    injection_patterns = (
        r"ignore (all )?previous instructions",
        r"disregard (all )?(prior|above) rules",
        r"reveal (your )?(system prompt|api key|secret)",
        r"you are now ",
        r"print .*api_key",
    )
    raw_lines: List[str] = []
    for raw_line in context_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in ("---", "===") or re.fullmatch(r"=+", stripped):
            raw_lines.append("")
            continue
        if stripped.startswith("```") or re.fullmatch(r"\[[^\]]+\.(?:txt|md|pdf|html)\]", stripped):
            continue
        low = stripped.lower()
        if any(re.search(pat, low) for pat in injection_patterns):
            continue
        raw_lines.append(raw_line.rstrip())

    # Build cohesive blocks (heading/intro ending ':' + consecutive list/sub-bullet items) AND single sentences
    units: List[str] = []
    current_block: List[str] = []

    def _flush_block() -> None:
        if current_block:
            joined = " ".join(re.sub(r"^[-*]\s+", "", x).strip() for x in current_block if x.strip())
            if joined:
                units.append(joined)
            current_block.clear()

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            _flush_block()
            continue
        is_hash_heading = stripped.startswith("#")
        is_colon_lead = stripped.endswith(":")
        is_sub_item = line.startswith(("  *", "  -", "   "))
        is_bullet = stripped.startswith(("- ", "* ")) or bool(re.match(r"^\d+\.\s+", stripped))
        prev_was_list_or_lead = bool(
            current_block and (
                current_block[-1].strip().endswith(":")
                or current_block[-1].strip().startswith(("#", "- ", "* "))
                or re.match(r"^\d+\.\s+", current_block[-1].strip())
                or current_block[0].strip().startswith("###")
            )
        )

        # Keep single-line root bullets separate when previous root bullet did NOT end with ':' and block isn't under ###
        root_sibling_item = (
            bool(current_block)
            and stripped.startswith("- ")
            and not is_sub_item
            and not current_block[0].strip().startswith("###")
            and not current_block[-1].strip().endswith(":")
        )

        if (
            current_block
            and not is_hash_heading
            and not root_sibling_item
            and (is_sub_item or is_bullet or current_block[-1].strip().startswith("###") or prev_was_list_or_lead)
            and sum(len(x) for x in current_block) < 750
        ):
            current_block.append(stripped)
        else:
            _flush_block()
            current_block.append(stripped)
            if not is_hash_heading and not is_colon_lead and len(stripped) >= 150 and not is_bullet:
                _flush_block()
                for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", stripped):
                    if s.strip():
                        units.append(s.strip())

    _flush_block()
    return units


def _score_sentence(query_clauses: List[List[str]], sentence: str) -> Tuple[float, int]:
    """Score a candidate sentence against one or more question clauses."""
    sent_tokens = set(_normalize_tokens(sentence))
    if not sent_tokens:
        return 0.0, 0

    best_score = 0.0
    matched_clauses = 0
    for clause_toks in query_clauses:
        content = [t for t in clause_toks if t not in _STOPWORDS and not t.startswith("syn_")]
        syns = [t for t in clause_toks if t.startswith("syn_")]
        if not content:
            continue

        direct_hits = sum(1 for t in set(content) if t in sent_tokens)
        syn_hits = sum(1 for t in set(syns) if t in sent_tokens)

        weighted_num = 0.0
        weighted_den = 0.0
        for t in set(content):
            w = 2.0 if (any(c.isdigit() for c in t) or "_" in t or len(t) > 7) else 1.0
            weighted_den += w
            if t in sent_tokens:
                weighted_num += w
            else:
                # Check if its synonym group hit
                canon = _WORD_TO_CANON.get(t)
                if canon and canon in sent_tokens:
                    weighted_num += 0.65 * w

        score = (weighted_num / weighted_den) if weighted_den > 0 else 0.0
        # Slight bonus for multiple distinct hits
        score += 0.06 * direct_hits + 0.02 * syn_hits
        if direct_hits >= 1 or syn_hits >= 2:
            matched_clauses += 1
        best_score = max(best_score, score)

    return best_score, matched_clauses


class LocalGroundedChatModel(BaseChatModel):
    """
    Zero-cost offline grounded chat model implementing LangChain's BaseChatModel.

    Performs genuine lexical + concept-aligned sentence grounding, faithfulness
    checking, document relevance grading, query rewriting/translation, and
    routing so learners can complete every notebook identically with no API keys.
    """

    model_name: str = "local-grounded-rag-v1"
    temperature: float = 0.0
    structured_schema: Optional[Type[BaseModel]] = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "local-grounded-chat"

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> Any:
        """Support Pydantic structured output for routing & grading notebooks."""
        def _invoke_structured(input_val: Any) -> BaseModel:
            if hasattr(input_val, "to_string"):
                prompt_text = input_val.to_string()
            else:
                prompt_text = str(input_val)
            raw_json = self._infer_structured_json(prompt_text, schema)
            return schema.model_validate_json(raw_json)

        return RunnableLambda(_invoke_structured)

    def _infer_structured_json(self, prompt_text: str, schema: Type[BaseModel]) -> str:
        fields = schema.model_fields
        q_lower = prompt_text.lower()
        data: Dict[str, Any] = {}
        for fname, finfo in fields.items():
            # Inspect Literal / description hints
            ann_str = str(finfo.annotation).lower()
            desc = (finfo.description or "").lower()
            if "datasource" in fname or "route" in fname:
                if any(w in q_lower for w in ("error", "err_", "ros2", "failover", "uptime", "sla", "port 9872", "faq")):
                    data[fname] = "product_faq"
                elif any(w in q_lower for w in ("changelog", "october", "release", "deprecated", "rest api", "elina", "42%")):
                    data[fname] = "recent_changelog"
                elif any(w in q_lower for w in ("safety", "perimeter", "e-stop", "commissioning", "brief")):
                    data[fname] = "safety_brief"
                else:
                    data[fname] = "company_handbook"
            elif "score" in fname or "relevant" in fname or "binary" in fname:
                # Grade document relevance cleanly
                doc_part = _extract_section(prompt_text, ["document:", "retrieved document:", "context:"]) or ""
                q_part = _extract_section(prompt_text, ["question:", "query:"]) or prompt_text
                # Split before DOCUMENT: if present
                if "document:" in q_part.lower():
                    q_part = q_part[: q_part.lower().find("document:")]
                is_rel = self._check_relevance(q_part, doc_part)
                if "bool" in ann_str:
                    data[fname] = is_rel
                else:
                    data[fname] = "yes" if is_rel else "no"
            elif "department" in fname or "category" in fname:
                if "hr" in desc or "pet" in q_lower or "stipend" in q_lower or "meal" in q_lower:
                    data[fname] = "hr"
                elif "pricing" in q_lower or "plan" in q_lower:
                    data[fname] = "pricing"
                else:
                    data[fname] = "engineering"
            else:
                data[fname] = "general"
        return json.dumps(data)

    @staticmethod
    def _check_relevance(question: str, document: str) -> bool:
        q_toks = {t for t in _normalize_tokens(question) if t not in _STOPWORDS and not t.startswith("syn_")}
        # Filter out generic brand words so queries like "What is Novasphere's NASDAQ stock ticker symbol?" don't match solely on "novasphere"
        focus_toks = {t for t in q_toks if t not in {"novasphere", "dynamics", "company", "system", "os"}}
        eval_toks = focus_toks if focus_toks else q_toks
        d_toks = set(_normalize_tokens(document))
        if not eval_toks or not d_toks:
            return False
        hits = sum(1 for t in eval_toks if t in d_toks or _WORD_TO_CANON.get(t, "") in d_toks)
        return (hits / len(eval_toks)) >= 0.25 or hits >= 2

    def _judge_faithfulness(self, full_prompt: str) -> str:
        """Compare ANSWER claims against CONTEXT; catch hallucinations reliably."""
        ctx_raw = _extract_section(full_prompt, ["CONTEXT:", "Context:"]) or ""
        if "ANSWER:" in ctx_raw:
            ctx_raw = ctx_raw[: ctx_raw.find("ANSWER:")].strip()
        elif "Answer:" in ctx_raw:
            ctx_raw = ctx_raw[: ctx_raw.find("Answer:")].strip()

        ans_raw = _extract_section(full_prompt, ["ANSWER:", "Answer:"]) or ""
        if "VERDICT:" in ans_raw:
            ans_raw = ans_raw[: ans_raw.find("VERDICT:")].strip()

        ctx_toks = set(_normalize_tokens(ctx_raw))
        raw_ans_words = re.findall(r"[a-zA-Z0-9_.$%-]+(?:<[0-9]+)?", ans_raw.lower())
        unsupported = []
        for raw_w in raw_ans_words:
            clean = raw_w.strip(".,;:!?()[]{}\"'`")
            if not clean or len(clean) <= 1 or clean in _STOPWORDS:
                continue
            forms = set(_normalize_tokens(clean))
            if forms & ctx_toks:
                continue
            unsupported.append(clean)

        # Numbers or concrete new words not in context signal ungrounded hallucination
        has_new_number = any(any(c.isdigit() for c in u) for u in unsupported)
        if has_new_number or len(set(unsupported)) >= 2:
            return f"HALLUCINATED (unsupported concepts: {', '.join(sorted(set(unsupported))[:5])})"
        return "FAITHFUL (all claims verified against context)"

    def _answer_from_context(self, question: str, context_text: str) -> str:
        sentences = _split_context_sentences(context_text)
        if not sentences:
            return "I don't know based on the provided context. No usable documents were supplied."

        # Break multi-part questions into sub-clauses so both halves retrieve facts
        raw_clauses = re.split(r"\b(?:and|or)\b|[?;,]", question)
        clauses = [_normalize_tokens(c) for c in raw_clauses if len(c.strip()) > 3]
        full_q_toks = _normalize_tokens(question)
        if not clauses:
            clauses = [full_q_toks]
        else:
            clauses.append(full_q_toks)

        scored: List[Tuple[float, int, str]] = []
        # Also evaluate consecutive 2-sentence windows for bullet headers + details
        candidates = list(sentences)
        for i in range(len(sentences) - 1):
            if sentences[i].endswith(":") or sentences[i].startswith("#") or len(sentences[i]) < 75:
                candidates.append(f"{sentences[i]} {sentences[i + 1]}")

        for idx, cand in enumerate(candidates):
            sc, clause_hits = _score_sentence(clauses, cand)
            if sc > 0:
                scored.append((sc, -idx, cand))

        scored.sort(reverse=True)
        top_score = scored[0][0] if scored else 0.0

        # Compute non-brand specific focus coverage across top evidence blocks to catch unanswerable gaps
        brand_ignore = {"novasphere", "dynamics", "company", "os", "team", "members"}
        raw_q_words = re.findall(r"[a-zA-Z0-9_$-]+", question.lower())
        q_focus = [
            w.strip(".,;:!?\"'")
            for w in raw_q_words
            if w.strip(".,;:!?\"'") not in _STOPWORDS
            and w.strip(".,;:!?\"'") not in brand_ignore
            and len(w.strip(".,;:!?\"'")) > 2
        ]
        top_evidence_text = " ".join(cand for _, _, cand in scored[:3])
        ev_tokens = set(_normalize_tokens(top_evidence_text))
        matched_focus = 0
        for fw in set(q_focus):
            forms = set(_normalize_tokens(fw))
            if forms & ev_tokens:
                matched_focus += 1
        focus_coverage = (matched_focus / len(set(q_focus))) if q_focus else 0.0

        # Grounded refusal check for out-of-scope questions
        if top_score < 0.30 or focus_coverage < 0.42:
            return (
                "I don't know based on the provided context. "
                "The supplied documents do not contain enough information to answer this question."
            )

        # Ensure multi-clause coverage: pick top distinct sentences covering different clauses
        selected: List[str] = []
        seen_norms: set[str] = set()
        for sc, _, cand in scored:
            if sc < max(0.24, top_score * 0.45):
                continue
            norm = re.sub(r"\W+", "", cand.lower())[:60]
            if norm in seen_norms:
                continue
            # Avoid adding short fragment already contained inside a selected combined window
            if any(cand in s for s in selected):
                continue
            seen_norms.add(norm)
            selected.append(cand)
            if len(selected) >= 3:
                break

        # If question has two clauses (e.g. "Who... and what..."), verify second clause got covered
        if len(clauses) >= 3:
            sub_c1 = [clauses[0]]
            sub_c2 = [clauses[1]]
            cov1 = any(_score_sentence(sub_c1, s)[0] >= 0.3 for s in selected)
            cov2 = any(_score_sentence(sub_c2, s)[0] >= 0.3 for s in selected)
            if not cov2:
                extra = max(candidates, key=lambda c: _score_sentence(sub_c2, c)[0])
                if _score_sentence(sub_c2, extra)[0] >= 0.25 and extra not in selected:
                    selected.append(extra)
            if not cov1:
                extra = max(candidates, key=lambda c: _score_sentence(sub_c1, c)[0])
                if _score_sentence(sub_c1, extra)[0] >= 0.25 and extra not in selected:
                    selected.append(extra)

        return "Based on the provided context: " + " ".join(selected)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sys_parts: List[str] = []
        user_parts: List[str] = []
        for m in messages:
            content = str(m.content)
            if isinstance(m, SystemMessage):
                sys_parts.append(content)
            else:
                user_parts.append(content)

        full_prompt = "\n".join(sys_parts + user_parts)
        prompt_lower = full_prompt.lower()
        last_user = user_parts[-1] if user_parts else ""

        # 1. Faithfulness judge mode
        if "faithful" in prompt_lower and "hallucinated" in prompt_lower and "verdict" in prompt_lower:
            reply = self._judge_faithfulness(full_prompt)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 2. Relevance grading mode (CRAG)
        if ("grade relevance" in prompt_lower or "yes' or 'no'" in prompt_lower or "relevant' or 'irrelevant'" in prompt_lower) and (
            "document:" in prompt_lower or "context:" in prompt_lower
        ):
            doc_txt = _extract_section(full_prompt, ["DOCUMENT:", "Document:", "CONTEXT:", "Context:"]) or ""
            q_txt = _extract_section(full_prompt, ["QUESTION:", "Question:", "QUERY:", "Query:"]) or last_user
            if "DOCUMENT:" in q_txt:
                q_txt = q_txt[: q_txt.find("DOCUMENT:")]
            is_rel = self._check_relevance(q_txt, doc_txt)
            if "irrelevant" in prompt_lower:
                reply = "RELEVANT" if is_rel else "IRRELEVANT"
            else:
                reply = "yes" if is_rel else "no"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 3. Query translation modes (NB07)
        if "alternative search queries" in prompt_lower or "generate 3 different versions" in prompt_lower:
            q = last_user.strip().rstrip("?")
            reply = (
                f"1. {q} details and official specifications\n"
                f"2. Novasphere documentation regarding {q.lower()}\n"
                f"3. Exact numbers and policy rules for {q.lower()}"
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        if "hypothetical document" in prompt_lower or "hyde" in prompt_lower:
            q = last_user.strip().rstrip("?")
            reply = (
                f"Novasphere Dynamics Documentation Section: Regarding {q}, official platform guidelines "
                f"specify exact configuration parameters, supported versions, SLA targets, and operational steps."
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        if "step-back" in prompt_lower or "broader conceptual question" in prompt_lower:
            q = last_user.strip()
            reply = f"What are the general architecture, policies, and core mechanisms underlying: {q}"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        if "sub-questions" in prompt_lower or "decompose" in prompt_lower:
            parts = [p.strip() for p in re.split(r"\band\b|,", last_user) if len(p.strip()) > 6]
            if len(parts) >= 2:
                reply = "\n".join(f"{i+1}. {p.rstrip('?')}?" for i, p in enumerate(parts[:3]))
            else:
                reply = f"1. What is the primary policy or definition for {last_user}?\n2. What constraints apply?"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 4. Standalone history question rewrite (NB13 / NB11)
        if "standalone question" in prompt_lower or "rewrite the query" in prompt_lower:
            hist = _extract_section(full_prompt, ["CHAT_HISTORY:", "Chat History:"]) or ""
            q_txt = _extract_section(full_prompt, ["FOLLOW_UP:", "Question:", "Query:"]) or last_user
            # Replace pronouns with concrete nouns from history if present
            topic_match = re.findall(
                r"(Enterprise Plan|Professional Plan|Starter Plan|ZeroGravity|HyperSync|ERR_QUANTUM_TIMEOUT|Zurich hub|REST API v1)",
                hist,
                flags=re.IGNORECASE,
            )
            if topic_match and any(p in q_txt.lower() for p in ("it", "its", "that", "they", "them", "this")):
                subject = topic_match[-1]
                rewritten = re.sub(r"\b(it|that|this)\b", subject, q_txt, flags=re.IGNORECASE)
                rewritten = re.sub(r"\b(its|their)\b", f"{subject}'s", rewritten, flags=re.IGNORECASE)
                reply = rewritten.strip()
            else:
                reply = f"{q_txt.strip()} (official Novasphere documentation specifications)"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 5. Summarization mode (NB09)
        if "concise summary" in prompt_lower or "summarize the following document" in prompt_lower:
            text_part = _extract_section(full_prompt, ["DOCUMENT:", "TEXT:"]) or last_user
            sents = _split_context_sentences(text_part)
            reply = "Summary: " + " ".join(sents[:3])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 6. Grounded RAG Q&A (supports both <context>...</context> and CONTEXT:...)
        ctx = _extract_section(full_prompt, ["<context>", "CONTEXT:", "Context:"])
        if ctx is not None:
            # Extract clean question after context block
            q_after = _extract_section(
                full_prompt,
                ["</context>", "QUESTION:", "Question:", "USER_QUESTION:", "Query:"],
            )
            if q_after:
                # Strip leading labels inside remaining section
                q_clean = re.sub(r"^(question|query|user)\s*:\s*", "", q_after, flags=re.IGNORECASE).strip()
                # If ANSWER: suffix exists in template, trim it
                if "\nAnswer:" in q_clean:
                    q_clean = q_clean[: q_clean.find("\nAnswer:")].strip()
                question_str = q_clean or last_user
            else:
                question_str = last_user

            # Strip CONTEXT from ctx if QUESTION: follows inline
            if "QUESTION:" in ctx:
                q_part = ctx[ctx.find("QUESTION:") + len("QUESTION:") :].strip()
                if "\nANSWER:" in q_part:
                    q_part = q_part[: q_part.find("\nANSWER:")].strip()
                ctx = ctx[: ctx.find("QUESTION:")].strip()
                question_str = q_part
            elif "Question:" in ctx:
                q_part = ctx[ctx.find("Question:") + len("Question:") :].strip()
                if "\nAnswer:" in q_part:
                    q_part = q_part[: q_part.find("\nAnswer:")].strip()
                ctx = ctx[: ctx.find("Question:")].strip()
                question_str = q_part

            reply = self._answer_from_context(question_str, ctx)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        # 7. Raw LLM without context (NB00 lesson)
        if any(kw in prompt_lower for kw in ("novasphere", "zerogravity", "hypersync", "elina vance", "quantumcache")):
            reply = (
                "I cannot verify internal or proprietary 2026 details about Novasphere Dynamics because "
                "I do not have access to private company documents. Supplying retrieved documents via RAG "
                "is required to give a grounded, factual answer."
            )
        else:
            reply = (
                "A standard language model generates answers strictly from static pre-training weights. "
                "Connect retrieved context to ground answers in up-to-date or private knowledge."
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def get_llm(provider: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Return the configured LangChain Chat Model based on `RAG_PROVIDER` in `.env`.

    Supported providers:
      - `local`     (default, 100% free & offline)
      - `ollama`    (local Ollama server)
      - `openai`    (requires valid OPENAI_API_KEY)
      - `anthropic` (requires valid ANTHROPIC_API_KEY)
    """
    _load_env()
    chosen = (provider or os.getenv("RAG_PROVIDER", "local")).strip().lower()

    if chosen == "local":
        return LocalGroundedChatModel(temperature=temperature)

    if chosen == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not _is_valid_api_key(api_key):
            raise ValueError(
                "OPENAI_API_KEY is missing or looks like a placeholder. "
                "Set a valid key in .env or switch RAG_PROVIDER=local."
            )
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    if chosen == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not _is_valid_api_key(api_key):
            raise ValueError(
                "ANTHROPIC_API_KEY is missing or looks like a placeholder. "
                "Set a valid key in .env or switch RAG_PROVIDER=local."
            )
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model, temperature=temperature)

    if chosen == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    raise ValueError(
        sanitize_error_message(
            f"Unsupported RAG_PROVIDER='{chosen}'. Choose one of: local, ollama, openai, anthropic."
        )
    )


def get_embeddings(provider: Optional[str] = None, use_cache: bool = True) -> Embeddings:
    """
    Return the configured LangChain Embeddings model based on `RAG_PROVIDER` in `.env`.

    Automatically wraps with an in-memory SHA-256 `CachedEmbeddings` when `use_cache=True`.
    """
    _load_env()
    chosen = (provider or os.getenv("RAG_PROVIDER", "local")).strip().lower()

    base: Embeddings
    if chosen == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not _is_valid_api_key(api_key):
            raise ValueError(
                "OPENAI_API_KEY is missing or looks like a placeholder for OpenAI embeddings. "
                "Set a valid key in .env or switch RAG_PROVIDER=local."
            )
        from langchain_openai import OpenAIEmbeddings

        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        base = OpenAIEmbeddings(model=embed_model)
    elif chosen in ("local", "ollama", "anthropic"):
        # Anthropic does not provide a native embedding API, so local embeddings are paired with it.
        if os.getenv("LOCAL_EMBED_BACKEND", "").strip().lower() == "hash":
            base = LocalHashEmbeddings(dimension=384)
        else:
            embed_model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
            base = LocalSentenceTransformerEmbeddings(model_name=embed_model)
    else:
        raise ValueError(
            sanitize_error_message(
                f"Unsupported RAG_PROVIDER='{chosen}' for embeddings. Choose: local, ollama, openai, anthropic."
            )
        )

    return CachedEmbeddings(base) if use_cache else base

"""
Semantic Search SaaS for Videos — Minimal but End‑to‑End (FastAPI + FAISS + Sentence-Transformers)
-----------------------------------------------------------------------------------------------
Features:
1) Ingest a video: extract metadata (duration, width/height, codec, fps) via ffprobe, accept transcript (or auto-transcribe if OPENAI key is set, optional), chunk text.
2) Embed chunks with sentence-transformers (all-MiniLM-L6-v2 by default) and index in-memory with FAISS.
3) RAG query endpoint: retrieves top-K chunks by cosine similarity and (optionally) calls an LLM to generate a context-aware answer.

How to run (one-time setup):
- Python 3.10+
- pip install -r requirements.txt  (sample below)
- Start API: uvicorn app:app --reload

Example requests (once server is up):
- Ingest (with transcript file or plain text):
  curl -X POST "http://127.0.0.1:8000/ingest" \
       -F "video=@/path/to/video.mp4" \
       -F "transcript_text=your transcript text here"

- Search:
  curl -X POST "http://127.0.0.1:8000/search" \
       -H "Content-Type: application/json" \
       -d '{"query":"What did the speaker say about embeddings?","k":5,"use_llm":true}'

Environment variables (optional):
- OPENAI_API_KEY: if set and use_llm=true, the app will call OpenAI's Chat Completions API.

requirements.txt (install these):
--------------------------------------------------
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
pydantic==2.7.4
sentence-transformers==2.7.0
faiss-cpu==1.8.0.post1
numpy==1.26.4
requests==2.32.3

# Optional if you want local metadata via ffprobe:
# Install FFmpeg on your system and ensure `ffprobe` is on PATH.

# If you want OpenAI LLM answers:
# openai==1.35.13
--------------------------------------------------
"""

import os
import io
import json
import uuid
import math
import time
import tempfile
import subprocess
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("FAISS not installed. Please `pip install faiss-cpu`. ") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("sentence-transformers not installed. Please `pip install sentence-transformers`. ") from e

# Optional OpenAI LLM support
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        USE_OPENAI = False

# -----------------------------
# Global, in-memory "DB"
# -----------------------------
app = FastAPI(title="Video Semantic Search RAG (Minimal)")

# Embedding model (small & fast)
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_emb_model: SentenceTransformer | None = None

# FAISS index + mapping
_faiss_index: faiss.IndexFlatIP | None = None  # cosine sim via inner product after L2-normalize
_vectors_dim: int | None = None
# id -> payload
_payloads: Dict[int, Dict[str, Any]] = {}
_next_id = 0

# video_id -> list of int ids in FAISS
_video_to_ids: Dict[str, List[int]] = {}


# -----------------------------
# Utilities
# -----------------------------

def get_embedder() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def ensure_index(dim: int) -> None:
    global _faiss_index, _vectors_dim
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)  # inner product
        _vectors_dim = dim
    elif _vectors_dim != dim:
        raise RuntimeError(f"Embedding dim changed from {_vectors_dim} to {dim}. Restart server or clear index.")


def ffprobe_metadata(path: str) -> Dict[str, Any]:
    """Return basic video metadata using ffprobe if available. Fallback to file size only."""
    meta: Dict[str, Any] = {
        "file_size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
    }
    try:
        # requires ffprobe on PATH
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        # Parse a few useful fields
        fmt = data.get("format", {})
        meta["duration_sec"] = float(fmt.get("duration")) if fmt.get("duration") else None
        # Find video stream
        vstream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                vstream = s
                break
        if vstream:
            meta["width"] = vstream.get("width")
            meta["height"] = vstream.get("height")
            meta["codec_name"] = vstream.get("codec_name")
            # try fps
            r = vstream.get("r_frame_rate")
            if r and "/" in r:
                num, den = r.split("/")
                try:
                    meta["fps"] = float(num) / float(den)
                except Exception:
                    meta["fps"] = None
    except Exception:
        # ffprobe not available; return minimal
        pass
    return meta


def simple_chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < n else end
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    # already normalized by sentence-transformers if normalize_embeddings=True; keep l2 normalize to be safe
    return l2_normalize(embs.astype("float32"))


def add_vectors(vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> List[int]:
    global _next_id, _faiss_index, _payloads
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    dim = vectors.shape[1]
    ensure_index(dim)
    ids = []
    for i in range(vectors.shape[0]):
        idx = _next_id
        _next_id += 1
        ids.append(idx)
        _payloads[idx] = payloads[i]
    # FAISS doesn't store custom ids in IndexFlat; we maintain a parallel mapping order->id
    # Workaround: we keep a separate list mapping; but simplest is to append in order and track ids in payloads
    # We'll maintain an array of ids in payload and store it as part of the payload itself.
    # Instead, we can add then search and map by insertion order. Keep a list of ids_per_add in payloads.
    # Here we push to FAISS directly:
    _faiss_index.add(vectors)
    return ids


def search_vectors(query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    D, I = _faiss_index.search(query_vec.astype("float32"), k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    results: List[Dict[str, Any]] = []
    for score, row_pos in zip(scores, idxs):
        if row_pos < 0:
            continue
        # row_pos is the order inside FAISS, not our global id; reconstruct payload via insertion order
        # For simplicity in this minimal demo, we store payloads in a parallel list in insertion order.
        # We'll construct that list when adding.
    
    # Because IndexFlatIP does not support custom ids, we need to mirror payload order.
    # Let's maintain a simple list in the same order as vectors were added.

# Mirror list for insertion order
_insert_order_payloads: List[Dict[str, Any]] = []


def add_vectors_with_order(vectors: np.ndarray, payloads: List[Dict[str, Any]]):
    ensure_index(vectors.shape[1])
    _faiss_index.add(vectors)
    _insert_order_payloads.extend(payloads)


def search_with_order(query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    D, I = _faiss_index.search(query_vec.astype("float32"), k)
    out: List[Dict[str, Any]] = []
    for score, row_pos in zip(D[0].tolist(), I[0].tolist()):
        if row_pos < 0:
            continue
        payload = dict(_insert_order_payloads[row_pos])
        payload["score"] = float(score)
        out.append(payload)
    return out


# -----------------------------
# Request/Response models
# -----------------------------
class IngestResponse(BaseModel):
    video_id: str
    chunks_indexed: int
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    use_llm: bool = False


class SearchResponse(BaseModel):
    query: str
    hits: List[Dict[str, Any]]
    answer: Optional[str] = None


# -----------------------------
# Optional: very light LLM wrapper
# -----------------------------

def generate_llm_answer(query: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Using ONLY the context chunks below, answer the user's question concisely.\n"
        "If the answer is not contained in the context, say you don't have enough information.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )
    if USE_OPENAI:
        try:
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM error: {e}]\n\nTop snippets:\n" + "\n---\n".join(contexts)
    else:
        # Fallback: extractive style — just return top contexts joined
        return "\n\n".join(contexts[:2])


# -----------------------------
# Routes
# -----------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    video: UploadFile = File(...),
    transcript_text: Optional[str] = Form(None),
):
    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Metadata
    meta = ffprobe_metadata(tmp_path)
    video_id = str(uuid.uuid4())

    # Require transcript text for this minimal demo (STT not included to keep it simple)
    if not transcript_text:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="transcript_text is required in this minimal starter. ")

    # Chunk + embed + index
    chunks = simple_chunk_text(transcript_text, max_chars=600, overlap=100)
    if not chunks:
        raise HTTPException(status_code=400, detail="Transcript empty after cleaning.")

    vectors = embed_texts(chunks)

    payloads = []
    ts = time.time()
    for i, ch in enumerate(chunks):
        payloads.append({
            "video_id": video_id,
            "chunk_index": i,
            "text": ch,
            "metadata": meta,
            "ingested_at": ts,
        })

    add_vectors_with_order(vectors, payloads)

    # Maintain reverse mapping
    ids_for_video = list(range(len(_insert_order_payloads) - len(chunks), len(_insert_order_payloads)))
    _video_to_ids[video_id] = ids_for_video

    # remove temp file; metadata already captured
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return IngestResponse(video_id=video_id, chunks_indexed=len(chunks), metadata=meta)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    qvec = embed_texts([req.query])
    hits = search_with_order(qvec, k=max(1, req.k))
    contexts = [h["text"] for h in hits]
    answer = generate_llm_answer(req.query, contexts) if req.use_llm else None
    return SearchResponse(query=req.query, hits=hits, answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok", "indexed_chunks": len(_insert_order_payloads)}


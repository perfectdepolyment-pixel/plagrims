"""
OriginCheck API - knowledge-based plagiarism checker
Model: all-MiniLM-L6-v2 via ONNX (fastembed) -> light, fast, CPU only.
"""
import io
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding

# ----------------------------------------------------------------------------
# Config (all overridable with environment variables)
# ----------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE = os.getenv("MODEL_CACHE", "/app/models")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
API_KEY = os.getenv("API_KEY", "")  # empty = auth disabled (dev only)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "origincheck.db")))
DIM = 384
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "3"))        # sentences per chunk
KB_STRIDE = int(os.getenv("KB_STRIDE", "2"))            # overlap for KB chunks
MAX_WINDOWS = int(os.getenv("MAX_WINDOWS", "3000"))     # cap per check (speed)
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "15"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "400000"))
MIN_SENT_CHARS = 20
MIN_WINDOW_CHARS = 40
LOW_MAX, MODERATE_MAX = 15, 40  # status bands (match the frontend)

COLORS = ["#FDE68A", "#BFDBFE", "#FBCFE8", "#BBF7D0", "#DDD6FE", "#FED7AA", "#A5F3FC", "#FECACA"]

model: Optional[TextEmbedding] = None


# ----------------------------------------------------------------------------
# Text extraction and chunking
# ----------------------------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        elif ext == "docx":
            from docx import Document

            doc = Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            text = data.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(415, "Unsupported file type. Use PDF, DOCX or TXT.")
    except HTTPException:
        raise
    except Exception as e:  # corrupted / encrypted files
        raise HTTPException(422, f"Could not read the file: {e}")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


REF_RE = re.compile(r"^\s*(references|reference list|bibliography|works cited)\s*:?\s*$", re.I | re.M)


def strip_references(text: str) -> str:
    matches = [m for m in REF_RE.finditer(text) if m.start() > len(text) * 0.5]
    return text[: matches[-1].start()].rstrip() if matches else text


SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


def split_sentences(text: str):
    out = []
    for m in SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        if e - s >= MIN_SENT_CHARS:
            out.append((s, e))
    return out


def make_windows(sents, size: int, stride: int):
    wins, i = [], 0
    while i < len(sents):
        grp = sents[i : i + size]
        s, e = grp[0][0], grp[-1][1]
        if e - s >= MIN_WINDOW_CHARS:
            wins.append((s, e))
        if i + size >= len(sents):
            break
        i += stride
    return wins


def embed(texts):
    vecs = np.array(list(model.embed([t[:1500] for t in texts], batch_size=64)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# ----------------------------------------------------------------------------
# Knowledge base: SQLite for storage, in-memory NumPy matrix for fast search
# ----------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    author      TEXT DEFAULT '',
    year        INTEGER,
    type        TEXT DEFAULT '',
    collection  TEXT DEFAULT 'General',
    chunks      INTEGER NOT NULL,
    chars       INTEGER NOT NULL,
    added_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text    TEXT NOT NULL,
    vector  BLOB NOT NULL            -- 384 x float32, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS reports (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    score       REAL NOT NULL,
    level       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    result      TEXT NOT NULL        -- full JSON report
);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at DESC);
"""


class KnowledgeBase:
    def __init__(self):
        self.lock = threading.RLock()
        self.db: Optional[sqlite3.Connection] = None
        self.vectors = np.zeros((0, DIM), dtype=np.float32)
        self.chunk_doc: list[str] = []
        self.chunk_text: list[str] = []
        self.docs: dict[str, dict] = {}

    @staticmethod
    def _connect(path: str) -> sqlite3.Connection:
        db = sqlite3.connect(path, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.executescript(SCHEMA)
        db.commit()
        return db

    def open(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.db = self._connect(str(DB_PATH))
        except (OSError, sqlite3.Error) as e:
            print(f"[warn] cannot open {DB_PATH} ({e}); falling back to an in-memory database")
            self.db = self._connect(":memory:")
        self._load_cache()

    def _load_cache(self):
        """Load all vectors into one matrix once at startup (search stays fast)."""
        with self.lock:
            self.docs = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM documents")}
            rows = self.db.execute("SELECT doc_id, text, vector FROM chunks ORDER BY id").fetchall()
            self.chunk_doc = [r["doc_id"] for r in rows]
            self.chunk_text = [r["text"] for r in rows]
            if rows:
                buf = b"".join(r["vector"] for r in rows)
                self.vectors = np.frombuffer(buf, dtype=np.float32).reshape(-1, DIM).copy()
            else:
                self.vectors = np.zeros((0, DIM), dtype=np.float32)

    # ---- documents -------------------------------------------------------
    def add(self, meta: dict, vecs: np.ndarray, texts: list[str]):
        with self.lock:
            with self.db:  # one transaction: commit on success, rollback on error
                self.db.execute(
                    "INSERT INTO documents (id, title, author, year, type, collection, chunks, chars, added_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        meta["id"], meta["title"], meta["author"], meta["year"], meta["type"],
                        meta["collection"], meta["chunks"], meta["chars"], meta["added_at"],
                    ),
                )
                self.db.executemany(
                    "INSERT INTO chunks (doc_id, text, vector) VALUES (?,?,?)",
                    [(meta["id"], t, np.ascontiguousarray(v, dtype=np.float32).tobytes()) for t, v in zip(texts, vecs)],
                )
            self.vectors = np.vstack([self.vectors, vecs])
            self.chunk_doc.extend([meta["id"]] * len(texts))
            self.chunk_text.extend(texts)
            self.docs[meta["id"]] = meta

    def remove(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            with self.db:
                self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))  # chunks cascade
            keep = [i for i, d in enumerate(self.chunk_doc) if d != doc_id]
            self.vectors = self.vectors[keep]
            self.chunk_doc = [self.chunk_doc[i] for i in keep]
            self.chunk_text = [self.chunk_text[i] for i in keep]
            del self.docs[doc_id]
            return True

    def snapshot(self):
        with self.lock:
            return self.vectors, list(self.chunk_doc), list(self.chunk_text), dict(self.docs)

    # ---- reports ---------------------------------------------------------
    def save_report(self, report_id: str, title: Optional[str], result: dict):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO reports (id, title, score, level, created_at, result) VALUES (?,?,?,?,?,?)",
                (report_id, title, result["overall_score"], result["status"]["level"], int(time.time()), json.dumps(result)),
            )

    def get_report(self, report_id: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT result FROM reports WHERE id=?", (report_id,)).fetchone()
        return json.loads(r["result"]) if r else None

    def list_reports(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT id, title, score, level, created_at FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_report(self, report_id: str) -> bool:
        with self.lock, self.db:
            return self.db.execute("DELETE FROM reports WHERE id=?", (report_id,)).rowcount > 0


kb = KnowledgeBase()


# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE)  # already baked into image
    list(model.embed(["warm up"]))
    kb.open()
    yield


app = FastAPI(title="OriginCheck API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "Invalid or missing X-API-Key header")


def read_upload(file: UploadFile) -> str:
    data = file.file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_FILE_MB} MB")
    return extract_text(file.filename or "", data)


def status_for(score: float) -> dict:
    if score <= LOW_MAX:
        return {"level": "low", "label": "Low similarity"}
    if score <= MODERATE_MAX:
        return {"level": "moderate", "label": "Moderate, review advised"}
    return {"level": "high", "label": "High, needs attention"}


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": model is not None, "kb_documents": len(kb.docs), "database": "sqlite"}


@app.post("/v1/kb/documents", dependencies=[Depends(require_key)])
def add_kb_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    title: str = Form(...),
    author: str = Form(""),
    year: Optional[int] = Form(None),
    doc_type: str = Form("Internal repository"),  # Internal repository | Reference book | Journal
    collection: str = Form("General"),
):
    """Add a document to the knowledge base (upload a file or send raw text)."""
    if file is None and not text:
        raise HTTPException(422, "Provide a file or text")
    content = read_upload(file) if file else clean_text(text)
    if len(content) > MAX_CHARS:
        raise HTTPException(413, "Document too long")

    sents = split_sentences(content)
    wins = make_windows(sents, WINDOW_SIZE, KB_STRIDE)
    if not wins:
        raise HTTPException(422, "Not enough readable text in the document")

    texts = [content[s:e] for s, e in wins]
    vecs = embed(texts)
    meta = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "author": author,
        "year": year,
        "type": doc_type,
        "collection": collection,
        "chunks": len(texts),
        "chars": len(content),
        "added_at": int(time.time()),
    }
    kb.add(meta, vecs, texts)
    return meta


@app.get("/v1/kb/documents", dependencies=[Depends(require_key)])
def list_kb_documents(collection: Optional[str] = None):
    docs = list(kb.docs.values())
    if collection:
        docs = [d for d in docs if d["collection"] == collection]
    return {"count": len(docs), "documents": sorted(docs, key=lambda d: -d["added_at"])}


@app.delete("/v1/kb/documents/{doc_id}", dependencies=[Depends(require_key)])
def delete_kb_document(doc_id: str):
    if not kb.remove(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


@app.post("/v1/check", dependencies=[Depends(require_key)])
def check(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    threshold: float = Form(DEFAULT_THRESHOLD),
    exclude_references: bool = Form(True),
    exclude_doc_id: Optional[str] = Form(None),  # skip a KB doc (e.g. the submission itself)
    collections: Optional[str] = Form(None),      # comma-separated collection names
    title: Optional[str] = Form(None),            # label for the saved report
    save: bool = Form(True),                      # store the report in SQLite
):
    """Compare a submission with the knowledge base and return a similarity report."""
    t0 = time.time()
    if file is None and not text:
        raise HTTPException(422, "Provide a file or text")
    content = read_upload(file) if file else clean_text(text)
    if len(content) > MAX_CHARS:
        raise HTTPException(413, "Document too long")
    if exclude_references:
        content = strip_references(content)

    sents = split_sentences(content)
    stride = 1
    wins = make_windows(sents, WINDOW_SIZE, stride)
    while len(wins) > MAX_WINDOWS and stride < WINDOW_SIZE:  # keep large docs fast
        stride += 1
        wins = make_windows(sents, WINDOW_SIZE, stride)
    if not wins:
        raise HTTPException(422, "Not enough readable text to analyse")

    vectors, chunk_doc, chunk_text, docs = kb.snapshot()
    allowed = {c.strip() for c in collections.split(",")} if collections else None
    valid = np.array(
        [
            (d != exclude_doc_id) and (allowed is None or docs.get(d, {}).get("collection") in allowed)
            for d in chunk_doc
        ],
        dtype=bool,
    )

    hits = []
    if len(chunk_doc) and valid.any():
        sub_vecs = embed([content[s:e] for s, e in wins])
        for b in range(0, len(wins), 64):  # batch to limit memory
            sims = sub_vecs[b : b + 64] @ vectors.T
            sims[:, ~valid] = -1.0
            best = sims.argmax(axis=1)
            for j, k in enumerate(best):
                score = float(sims[j, k])
                if score >= threshold:
                    s, e = wins[b + j]
                    hits.append(
                        {"start": s, "end": e, "source": chunk_doc[k], "sim": score, "snippet": chunk_text[k][:400]}
                    )

    # Merge overlapping hits into non-overlapping spans
    hits.sort(key=lambda h: (h["start"], h["end"]))
    spans = []
    for h in hits:
        if spans and h["start"] < spans[-1]["end"]:
            last = spans[-1]
            if h["source"] == last["source"]:
                last["end"] = max(last["end"], h["end"])
                if h["sim"] > last["sim"]:
                    last["sim"], last["snippet"] = h["sim"], h["snippet"]
            elif h["end"] > last["end"]:
                spans.append({**h, "start": last["end"]})
        else:
            spans.append(dict(h))

    total = max(len(content), 1)
    matched_chars = sum(s["end"] - s["start"] for s in spans)
    overall = round(min(100.0, matched_chars / total * 100), 1)

    # Per-source summary
    per_src: dict[str, dict] = {}
    for s in spans:
        p = per_src.setdefault(s["source"], {"chars": 0, "max_sim": 0.0, "passages": 0})
        p["chars"] += s["end"] - s["start"]
        p["max_sim"] = max(p["max_sim"], s["sim"])
        p["passages"] += 1
    ranked = sorted(per_src.items(), key=lambda kv: -kv[1]["chars"])
    sources = []
    color_of = {}
    for i, (sid, p) in enumerate(ranked):
        color_of[sid] = COLORS[i % len(COLORS)]
        d = docs.get(sid, {})
        sources.append(
            {
                "id": sid,
                "title": d.get("title", "Unknown"),
                "author": d.get("author", ""),
                "year": d.get("year"),
                "type": d.get("type", ""),
                "collection": d.get("collection", ""),
                "match_percent": round(p["chars"] / total * 100, 1),
                "max_similarity": round(p["max_sim"], 3),
                "matched_passages": p["passages"],
                "color": color_of[sid],
            }
        )

    result = {
        "report_id": None,
        "overall_score": overall,
        "status": status_for(overall),
        "word_count": len(content.split()),
        "threshold": threshold,
        "text": content,  # analysed text; span offsets refer to this string
        "sources": sources,
        "spans": [
            {
                "start": s["start"],
                "end": s["end"],
                "source_id": s["source"],
                "similarity": round(s["sim"], 3),
                "color": color_of[s["source"]],
                "source_snippet": s["snippet"],
            }
            for s in spans
        ],
        "processing_ms": int((time.time() - t0) * 1000),
    }
    if save:
        result["report_id"] = uuid.uuid4().hex[:12]
        kb.save_report(result["report_id"], title or (file.filename if file else None), result)
    return result


@app.get("/v1/reports", dependencies=[Depends(require_key)])
def list_reports(limit: int = 50):
    """Recent saved reports (for the submission history pages)."""
    return {"reports": kb.list_reports(max(1, min(limit, 200)))}


@app.get("/v1/reports/{report_id}", dependencies=[Depends(require_key)])
def get_report(report_id: str):
    report = kb.get_report(report_id)
    if report is None:
        raise HTTPException(404, "Report not found")
    return report


@app.delete("/v1/reports/{report_id}", dependencies=[Depends(require_key)])
def delete_report(report_id: str):
    if not kb.delete_report(report_id):
        raise HTTPException(404, "Report not found")
    return {"deleted": report_id}

import os
import uuid
import logging
from datetime import datetime
from typing import AsyncGenerator

import numpy as np
import faiss
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {chunks: [...], index: faiss.Index}
sessions: dict[str, dict] = {}

# Load embedding model once at startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def parse_docx(data: bytes) -> str:
    import io
    docx = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in docx.paragraphs if p.text.strip())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # IndexFlatIP = inner product; on L2-normalised vectors this equals cosine similarity
    vectors = embeddings.astype(np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF or DOCX.")

    data = await file.read()
    logger.info("[%s] Upload: %s", datetime.now().isoformat(), filename)

    try:
        text = parse_pdf(data) if ext == "pdf" else parse_docx(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chunks": chunks,
        "index": build_faiss_index(np.array(embeddings)),
        "filename": filename,
    }

    logger.info("[%s] Indexed %d chunks for session %s", datetime.now().isoformat(), len(chunks), session_id)
    return {"session_id": session_id, "chunk_count": len(chunks), "filename": filename}


class ChatRequest(BaseModel):
    session_id: str
    query: str
    temperature: float = 0.3


# ~100k tokens worth of words — safe limit for gpt-4o-mini's 128k context window
FULL_DOC_WORD_LIMIT = 75_000


async def stream_chat(session: dict, query: str, temperature: float) -> AsyncGenerator[str, None]:
    chunks = session["chunks"]
    total_words = sum(len(c.split()) for c in chunks)

    if total_words <= FULL_DOC_WORD_LIMIT:
        # Document fits in context — pass everything so the LLM can answer any question
        context_chunks = chunks
    else:
        # Large document: use FAISS to retrieve the most relevant 15 chunks
        query_vec = embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        _, top_indices = session["index"].search(query_vec, k=min(15, len(chunks)))
        context_chunks = [chunks[i] for i in top_indices[0]]

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a document assistant. Answer the user's question using ONLY the context "
        "excerpts provided below. Do NOT use any outside knowledge.\n"
        "If the answer cannot be found in the provided context, respond with exactly: "
        "\"This information is not available in the uploaded document.\"\n\n"
        f"CONTEXT:\n{context}"
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {delta}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    temperature = max(0.0, min(1.0, req.temperature))
    logger.info("[%s] Query (session %s): %s", datetime.now().isoformat(), req.session_id, req.query[:80])

    session = sessions[req.session_id]
    return StreamingResponse(
        stream_chat(session, req.query, temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    logger.info("[%s] Session deleted: %s", datetime.now().isoformat(), session_id)
    return {"status": "deleted"}


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

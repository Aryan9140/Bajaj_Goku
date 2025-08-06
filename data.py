# api.py — FastAPI + Pinecone Hybrid Search + Robust OCR + Groq→Groq→Mistral fallback
# Fixes: Pinecone 400 "Sparse vector must contain at least one value" by filtering empty chunks.
# Run:
#   pip install fastapi uvicorn python-dotenv httpx pymupdf pillow pytesseract \
#               pinecone pinecone-text langchain-community langchain-huggingface rapidfuzz
#   set/export: API_TOKEN, GROQ_API_KEY (or GROQ_API_KEY1), MISTRAL_API_KEY, PINECONE_API_KEY, HF_TOKEN(optional)
#   python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import os, re, io, json, time, asyncio, hashlib, tempfile
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import httpx, fitz
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from PIL import Image
from rapidfuzz import fuzz

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

load_dotenv()

API_TOKEN        = os.getenv("API_TOKEN", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY1", "")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "hybrid-search-langchain-pinecone")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")

PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")                       # Groq
SECONDARY_MODEL  = os.getenv("SECONDARY_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")  # Groq
MISTRAL_MODEL    = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

OCR_ENABLED      = os.getenv("OCR_ENABLED", "1") == "1"

# knobs
CTX_MAX_CHARS            = int(os.getenv("CTX_MAX_CHARS", "16000"))
TOPK_DEFAULT             = int(os.getenv("TOPK_DEFAULT", "12"))
TOPK_DEFINITION          = int(os.getenv("TOPK_DEFINITION", "18"))
SUSPECT_MIN_CHARS        = int(os.getenv("SUSPECT_MIN_CHARS", "40"))
LLM_TEMPERATURE          = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS           = int(os.getenv("LLM_MAX_TOKENS", "320"))
CONCURRENCY_PER_PDF      = int(os.getenv("CONCURRENCY_PER_PDF", "6"))
HTTP_TIMEOUT_SEC         = float(os.getenv("HTTP_TIMEOUT_SEC", "60"))
PDF_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("PDF_DOWNLOAD_TIMEOUT_SEC", "180"))

# OCR escalation thresholds
SHORT_PAGE_RATIO_OCR_ALL = float(os.getenv("SHORT_PAGE_RATIO_OCR_ALL", "0.5"))  # if >50% pages very short => OCR them pre-chunk

app = FastAPI(title="HackRX PDF Q&A (Pinecone Hybrid + Robust OCR + Groq→Mistral)")

class SingleRun(BaseModel):
    document: str
    questions: List[str]

PROMPT = """You are an expert policy analyst. Use ONLY the context to answer.
Copy all critical details EXACTLY (dates, months/years, waiting periods, percentages, limits/caps, counts, named Acts, eligibility and exclusions).
If Yes/No, say Yes/No first and immediately include the qualifying conditions.
For definitions (e.g., “Hospital”), provide the full formal definition.
If absent, reply exactly: Not mentioned in the policy.
Write 1–3 formal sentences in the style of the document. Do NOT invent anything.

Context:
{context}

Question:
{question}

Answer:"""

def _auth_ok(authorization: Optional[str]) -> bool:
    if not API_TOKEN or not authorization: return False
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_TOKEN
    return authorization.strip() == API_TOKEN

def _is_definition_query(q: str) -> bool:
    ql = q.lower()
    keys = ["define", "definition", "means", "hospital", "ayush",
            "maternity", "organ donor", "room rent", "icu", "sub-limit",
            "waiting period", "no claim discount", "ncd", "pre-existing", "ped"]
    return any(k in ql for k in keys)

_SENT_SPLIT = re.compile(r"(?<=[.?!:;])\s+(?=(?:[A-Z(]|\d+[.)\]]))")
_TOKEN_RE   = re.compile(r"[A-Za-z0-9]{2,}")

def split_sentences(text: str) -> List[str]:
    flat = (text or "").replace("\r", " ").replace("\n", " ")
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]

def token_count(txt: str) -> int:
    return len(_TOKEN_RE.findall(txt or ""))

def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def _ocr_image_from_pixmap(pix) -> str:
    if not OCR_ENABLED:
        return ""
    try:
        import pytesseract
    except Exception:
        return ""
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def selective_ocr_pages(pdf_path: str, pages: List[int], dpi: int = 200) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not OCR_ENABLED or not pages:
        return out
    try:
        doc = fitz.open(pdf_path)
        for p in pages:
            if p < 0 or p >= len(doc): continue
            try:
                pm = doc[p].get_pixmap(dpi=dpi)
                txt = _ocr_image_from_pixmap(pm).strip()
                if token_count(txt) >= 5:  # only accept meaningful OCR
                    out[p] = txt
            except Exception:
                pass
        doc.close()
    except Exception:
        pass
    return out

@dataclass
class IngestedDoc:
    namespace: str
    chunks: List[str]
    chunk_meta: List[Dict[str, Any]]
    bm25: BM25Encoder
    pdf_path: str

_ingest_cache: Dict[str, IngestedDoc] = {}

async def download_pdf_to_temp(url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        async with httpx.AsyncClient(timeout=PDF_DOWNLOAD_TIMEOUT_SEC) as cli:
            async with cli.stream("GET", url) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    tmp.write(chunk)
        return tmp.name

def extract_pages_text(path: str) -> List[str]:
    doc = fitz.open(path)
    out = []
    for page in doc:
        txt = page.get_text("text") or ""
        out.append(txt)
    doc.close()
    return out

def chunk_pages(pages: List[str], max_chars: int = 600) -> (List[str], List[Dict[str, Any]]):
    chunks, meta = [], []
    for pi, ptxt in enumerate(pages):
        sents = split_sentences(ptxt)
        buf, sent_start, chunk_id = "", 0, 0
        i = 0
        while i < len(sents):
            if not buf:
                buf = sents[i]; sent_start = i; i += 1
            else:
                if len(buf) + 1 + len(sents[i]) <= max_chars:
                    buf = f"{buf} {sents[i]}"; i += 1
                else:
                    chunks.append(buf)
                    meta.append({"page": pi, "chunk_id": chunk_id, "sent_start": sent_start})
                    chunk_id += 1
                    buf = ""
        if buf:
            chunks.append(buf)
            meta.append({"page": pi, "chunk_id": chunk_id, "sent_start": sent_start})
    return chunks, meta

def init_pinecone() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc

PC = init_pinecone()
PINDEX = PC.Index(PINECONE_INDEX)
EMB = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

async def ensure_ingested(doc_url: str) -> IngestedDoc:
    ns = f"doc_{short_hash(doc_url)}"
    if ns in _ingest_cache:
        return _ingest_cache[ns]

    pdf_path = await download_pdf_to_temp(doc_url)
    pages = extract_pages_text(pdf_path)

    # If many pages are "short", OCR them pre-chunk
    lens = [len((t or "").strip()) for t in pages]
    short_pages = [i for i, L in enumerate(lens) if L < SUSPECT_MIN_CHARS]
    if pages and (len(short_pages) / max(1, len(pages))) > SHORT_PAGE_RATIO_OCR_ALL:
        ocr_map = selective_ocr_pages(pdf_path, short_pages, dpi=200)
        if ocr_map:
            for i in short_pages:
                if i in ocr_map:
                    pages[i] = ocr_map[i]

    chunks, meta = chunk_pages(pages)

    # Filter empty/near-empty chunks to avoid empty sparse vectors
    filtered_chunks = []
    filtered_meta = []
    for c, m in zip(chunks, meta):
        if token_count(c) >= 5:
            filtered_chunks.append(c)
            filtered_meta.append(m)

    # If nothing meaningful, try OCR all pages once more
    if not filtered_chunks:
        ocr_map = selective_ocr_pages(pdf_path, list(range(len(pages))), dpi=200)
        if ocr_map:
            for i in range(len(pages)):
                if i in ocr_map and token_count(pages[i]) < 5:
                    pages[i] = ocr_map[i]
            chunks, meta = chunk_pages(pages)
            filtered_chunks, filtered_meta = [], []
            for c, m in zip(chunks, meta):
                if token_count(c) >= 5:
                    filtered_chunks.append(c)
                    filtered_meta.append(m)

    # Last resort: if still empty, create a minimal placeholder (prevents Pinecone 400)
    if not filtered_chunks:
        filtered_chunks = ["placeholder content (no extractable text)"]
        filtered_meta = [{"page": 0, "chunk_id": 0, "sent_start": 0}]

    bm25 = BM25Encoder().default()
    # Fit on whatever we have (always non-empty now)
    bm25.fit(filtered_chunks)

    # Upsert only the good chunks
    try:
        stats = PINDEX.describe_index_stats()
        present = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
    except Exception:
        present = 0

    if present < len(filtered_chunks):
        retriever = PineconeHybridSearchRetriever(
            embeddings=EMB, sparse_encoder=bm25, index=PINDEX, namespace=ns
        )
        BATCH = 128
        for i in range(0, len(filtered_chunks), BATCH):
            batch = filtered_chunks[i:i+BATCH]
            batch_meta = [{"page": filtered_meta[i+j]["page"], "chunk_id": filtered_meta[i+j]["chunk_id"]} for j in range(len(batch))]
            # IMPORTANT: filter again before add_texts, to be extra safe
            safe_batch = []
            safe_meta  = []
            for txt, md in zip(batch, batch_meta):
                if token_count(txt) >= 5:
                    safe_batch.append(txt)
                    safe_meta.append(md)
            if safe_batch:
                retriever.add_texts(texts=safe_batch, metadatas=safe_meta)

    ing = IngestedDoc(namespace=ns, chunks=filtered_chunks, chunk_meta=filtered_meta, bm25=bm25, pdf_path=pdf_path)
    _ingest_cache[ns] = ing
    return ing

def build_window_for_question(ing: IngestedDoc, question: str, is_def: bool) -> str:
    topk = TOPK_DEFINITION if is_def else TOPK_DEFAULT
    retriever = PineconeHybridSearchRetriever(
        embeddings=EMB, sparse_encoder=ing.bm25, index=PINDEX, namespace=ing.namespace
    )
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception:
        docs = retriever.invoke(question)

    scored = []
    for d in docs:
        try:
            s = float(d.metadata.get("score", 0.0))
        except Exception:
            s = 0.0
        scored.append((s, d))
    scored.sort(key=lambda t: t[0], reverse=True)

    parts: List[str] = []
    ocr_needed_pages = []

    for _, d in scored[:topk]:
        txt = (d.page_content or "").strip()
        page = d.metadata.get("page", None)
        if page is not None and token_count(txt) < 5:
            ocr_needed_pages.append(int(page))
            parts.append(txt)  # placeholder
        else:
            parts.append(txt)

    if ocr_needed_pages:
        ocr_map = selective_ocr_pages(ing.pdf_path, list(set(ocr_needed_pages)), dpi=200)
        for i, (_, d) in enumerate(scored[:topk]):
            page = d.metadata.get("page", None)
            if page is not None and token_count(parts[i]) < 5 and page in ocr_map:
                parts[i] = ocr_map[page]

    # If still empty, fuzzy fallback over local chunks
    if not any(token_count(p) >= 5 for p in parts):
        candidates = sorted(
            ((fuzz.token_set_ratio(question, c), c) for c in ing.chunks),
            key=lambda t: t[0], reverse=True
        )[:topk]
        parts = [c for _, c in candidates]

    ctx = " ".join([p for p in parts if p]).strip()
    if len(ctx) > CTX_MAX_CHARS:
        ctx = ctx[:CTX_MAX_CHARS]
    return ctx

async def _groq_chat(model: str, prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model, "temperature": LLM_TEMPERATURE, "top_p": 1,
        "max_tokens": LLM_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        if r.status_code in (429, 500, 502, 503, 504, 413):
            raise httpx.HTTPStatusError("Groq status", request=None, response=r)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def _mistral_chat(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY missing")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL, "temperature": LLM_TEMPERATURE, "top_p": 1,
        "max_tokens": LLM_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        if r.status_code in (429, 500, 502, 503, 504, 413):
            raise httpx.HTTPStatusError("Mistral status", request=None, response=r)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def ask_llm_with_fallbacks(prompt: str) -> str:
    ctx = prompt
    shrink_attempts = 0

    async def try_chain(p: str) -> str:
        delay = 0.25
        # Primary
        for _ in range(3):
            try:
                return await _groq_chat(PRIMARY_MODEL, p)
            except httpx.HTTPStatusError as e:
                if getattr(e, "response", None) and e.response.status_code == 413:
                    raise
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
            except Exception:
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
        # Secondary
        delay = 0.25
        for _ in range(3):
            try:
                return await _groq_chat(SECONDARY_MODEL, p)
            except httpx.HTTPStatusError as e:
                if getattr(e, "response", None) and e.response.status_code == 413:
                    raise
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
            except Exception:
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
        # Tertiary
        delay = 0.25
        for _ in range(3):
            try:
                return await _mistral_chat(p)
            except httpx.HTTPStatusError as e:
                if getattr(e, "response", None) and e.response.status_code == 413:
                    raise
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
            except Exception:
                await asyncio.sleep(delay); delay = min(1.0, delay * 2)
        raise RuntimeError("All model calls failed")

    while True:
        try:
            return await try_chain(ctx)
        except httpx.HTTPStatusError as e:
            if getattr(e, "response", None) and e.response.status_code == 413 and shrink_attempts < 2:
                shrink_attempts += 1
                m = re.search(r"Context:\n(.*)\n\nQuestion:", ctx, re.S)
                if m:
                    ctx_block = m.group(1)
                    L = len(ctx_block)
                    cut_start, cut_end = L // 3, 2 * L // 3
                    new_block = ctx_block[:cut_start] + " … " + ctx_block[cut_end:]
                    ctx = ctx.replace(ctx_block, new_block)
                else:
                    ctx = ctx[: int(len(ctx) * 0.65)]
                continue
            raise

async def answer_pdf(document: str, questions: List[str]) -> Dict[str, Any]:
    t_start = time.perf_counter()
    ing = await ensure_ingested(document)

    sem = asyncio.Semaphore(CONCURRENCY_PER_PDF)

    async def solve(q: str) -> str:
        async with sem:
            try:
                ctx = build_window_for_question(ing, q, is_def=_is_definition_query(q))
                prompt = PROMPT.format(context=ctx, question=q)
                ans = await ask_llm_with_fallbacks(prompt)
                # Trim accidental duplicate "Not mentioned..." if model added extra text
                nm = "Not mentioned in the policy."
                if nm in ans and len(ans.strip()) > len(nm) + 2:
                    # if there's other content + this phrase, drop the phrase
                    ans = ans.replace(nm, "").strip().rstrip(".")
                return ans.strip()
            except Exception as e:
                return f"ERROR: {e}"

    tasks = [asyncio.create_task(solve(q)) for q in questions]
    answers = await asyncio.gather(*tasks)

    elapsed = int((time.perf_counter() - t_start) * 1000)
    return {"answers": answers, "duration_ms": elapsed}

@app.get("/")
async def health():
    return {
        "ok": True,
        "index": PINECONE_INDEX,
        "ocr": OCR_ENABLED,
        "groq": bool(GROQ_API_KEY),
        "mistral": bool(MISTRAL_API_KEY),
    }

@app.post("/api/v1/hackrx/run")
async def run_single(body: SingleRun, authorization: Optional[str] = Header(default=None)):
    if not _auth_ok(authorization):
        raise HTTPException(401, "Unauthorized")
    if not body.document or not body.questions:
        raise HTTPException(400, "Provide 'document' and 'questions'.")
    try:
        return await answer_pdf(body.document, body.questions)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Upstream HTTP error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")

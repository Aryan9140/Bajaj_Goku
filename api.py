# test1.py — FastAPI: PDF-wise answers + per-question/per-PDF timing + Groq→Mistral fallback
# Run: python -m uvicorn test1:app --host 0.0.0.0 --port 8000
# Env:
#   API_TOKEN=...
#   GROQ_API_KEY or GROQ_API_KEY1=...
#   MISTRAL_API_KEY=...
#   OCR_ENABLED=1/0 (default 1)
#   GROQ_MODEL (default: meta-llama/llama-4-scout-17b-16e-instruct)
#   MISTRAL_MODEL (default: mistral-small-latest)

import os, re, io, json, time, asyncio, tempfile
from typing import List, Dict, Any, Optional
import httpx, fitz
from dotenv import load_dotenv
from PIL import Image
from rapidfuzz import fuzz

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# ── ENV ──────────────────────────────────────────────────────────────────────
load_dotenv()
API_TOKEN       = os.getenv("API_TOKEN")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OCR_ENABLED     = os.getenv("OCR_ENABLED", "1") == "1"
MODEL_GROQ      = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MODEL_MISTRAL   = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

app = FastAPI(title="Grouped PDF Q&A API (retrieval + Groq→Mistral fallback)")

# ── PROMPT: full, clause-rich answers ───────────────────────────────────────
PROMPT = """You are an expert document analyst. Use ONLY the following document context to answer the question.
Requirements:
- Copy all critical details EXACTLY (dates, months/years, waiting periods, limits/percents, caps, counts, named Acts, eligibility/conditions).
- If the question is Yes/No, say Yes/No and immediately include the qualifying conditions from the context.
- If it’s a definition (e.g., “Hospital”), provide the complete formal definition from the context.
- Write 1–3 full sentences in the formal style of the document. Do NOT give a one-word answer.
- If the information is absent, reply exactly: Not mentioned in the policy.

Context:
{context}

Question:
{question}

Answer:"""

# ── TEXT UTILS ───────────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[.?!:;])\s+(?=(?:[A-Z(]|\d+[.)\]]))")

def split_sentences(text: str) -> List[str]:
    flat = text.replace("\r", " ").replace("\n", " ")
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]

def needs_bigger_window(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "define", "definition", "means", "hospital", "ayush", "maternity",
        "organ donor", "room rent", "icu", "sub-limit", "waiting period",
        "no claim discount", "ncd", "pre-existing", "ped"
    ])

def build_window(corpus: str, question: str,
                 top_k: int = 22, lead: int = 2, tail: int = 4, max_chars: int = 16000) -> str:
    if needs_bigger_window(question):
        top_k, lead, tail, max_chars = 28, 3, 5, 22000
    sents = split_sentences(corpus)
    scored = sorted(((fuzz.token_set_ratio(question, s), i) for i, s in enumerate(sents)),
                    key=lambda t: t[0], reverse=True)[:200]
    keep = set()
    for _, i in scored[:top_k]:
        keep.add(i)
        for d in range(1, lead+1):
            if i-d >= 0: keep.add(i-d)
        for d in range(1, tail+1):
            if i+d < len(sents): keep.add(i+d)
    window = " ".join(sents[i] for i in sorted(keep))
    return window[:max_chars]

# ── PDF LOADING w/ Selective OCR ─────────────────────────────────────────────
def _ocr_image(pix) -> str:
    if not OCR_ENABLED:
        return ""
    try:
        import pytesseract
    except Exception:
        return ""
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)

def extract_pdf_text(url: str, ocr_min_len: int = 40, dpi: int = 200) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        with httpx.stream("GET", url, timeout=180.0) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(16384):
                tmp.write(chunk)
        path = tmp.name
    out = []
    try:
        doc = fitz.open(path)
        for page in doc:
            txt = page.get_text("text") or ""
            if len(txt.strip()) < ocr_min_len:
                try:
                    pm = page.get_pixmap(dpi=dpi)
                    txt_ocr = _ocr_image(pm)
                    if txt_ocr and len(txt_ocr.strip()) > len(txt.strip()):
                        txt = (txt + "\n" + txt_ocr).strip()
                except Exception:
                    pass
            out.append(txt)
        doc.close()
    finally:
        try: os.remove(path)
        except Exception: pass
    return "\n".join(out).strip()

# ── LLM CLIENTS (Groq primary → Mistral fallback) ───────────────────────────
async def call_groq(prompt: str, max_tokens: int = 700, temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_GROQ,
        "temperature": temperature, "top_p": 1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60.0) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        if r.status_code in (429, 413, 500, 502, 503):
            raise httpx.HTTPStatusError("Groq busy", request=None, response=r)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def call_mistral(prompt: str, max_tokens: int = 700, temperature: float = 0.2) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY missing")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_MISTRAL,
        "temperature": temperature, "top_p": 1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60.0) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        if r.status_code in (429, 413, 500, 502, 503):
            raise httpx.HTTPStatusError("Mistral busy", request=None, response=r)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def ask_llm(prompt: str) -> str:
    try:
        return await call_groq(prompt)
    except Exception:
        await asyncio.sleep(0.4)
        return await call_mistral(prompt)

# ── SCHEMAS ─────────────────────────────────────────────────────────────────
class Batch(BaseModel):
    document: str
    questions: List[str]

class RunBody(BaseModel):
    batches: List[Batch] = Field(default_factory=list)

def _auth_ok(authorization: Optional[str]) -> bool:
    if not API_TOKEN: return False
    if not authorization: return False
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_TOKEN
    return authorization.strip() == API_TOKEN

# ── CORE ─────────────────────────────────────────────────────────────────────
async def answer_questions_for_doc(doc_url: str, questions: List[str]) -> Dict[str, Any]:
    pdf_start = time.perf_counter()
    corpus = extract_pdf_text(doc_url)
    qa = []
    for q in questions:
        q_start = time.perf_counter()
        ctx = build_window(corpus, q)
        prompt = PROMPT.format(context=ctx, question=q)
        try:
            ans = await ask_llm(prompt)
        except Exception as e:
            ans = f"ERROR: {e}"
        q_ms = int((time.perf_counter() - q_start) * 1000)
        qa.append({"question": q, "answer": ans, "duration_ms": q_ms})
        await asyncio.sleep(0.05)
    pdf_ms = int((time.perf_counter() - pdf_start) * 1000)
    # answers-only array to match "pdf wise answers" testing
    answers_only = [x["answer"] for x in qa]
    return {
        "document": doc_url,
        "answers": answers_only,      # <- simple array
        "qa": qa,                     # <- detailed per question
        "duration_ms": pdf_ms
    }

# ── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"ok": True, "ocr": OCR_ENABLED, "groq": bool(GROQ_API_KEY), "mistral": bool(MISTRAL_API_KEY)}

@app.post("/api/v1/hackrx/run_grouped")
async def run_grouped(body: RunBody, authorization: Optional[str] = Header(default=None)):
    if not _auth_ok(authorization):
        raise HTTPException(401, "Unauthorized")
    if not body.batches:
        raise HTTPException(400, "Provide 'batches': [{document, questions}, ...]")

    all_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    # sequential to avoid rate limits; you can parallelize with a semaphore if needed
    for b in body.batches:
        try:
            results.append(await answer_questions_for_doc(b.document, b.questions))
        except Exception as e:
            # hard error on this PDF → return error shells for each question
            results.append({
                "document": b.document,
                "answers": [f"ERROR: {e}" for _ in b.questions],
                "qa": [{"question": q, "answer": f"ERROR: {e}", "duration_ms": 0} for q in b.questions],
                "duration_ms": 0
            })
    total_ms = int((time.perf_counter() - all_start) * 1000)
    return {"results": results, "total_duration_ms": total_ms}

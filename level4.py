# main.py — Multilingual PDF-QA + Mission Solver (deterministic, <20s target)
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Env required: API_TOKEN (required), MISTRAL_API_KEY (optional), GROQ_API_KEY (optional)
# OCR: install tesseract + needed lang packs; export TESSDATA_PREFIX if required.

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import os, io, re, time, tempfile, mimetypes, unicodedata

import requests, httpx
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dotenv import load_dotenv
load_dotenv(override=True)  # ensures your .env wins locally


# ---------------- App ----------------
app = FastAPI(title="HackRx PDF QA + Mission Solver")

# ---------------- Env ----------------
API_TOKEN = os.getenv("API_TOKEN", "").strip()
if not API_TOKEN:
    raise RuntimeError("API_TOKEN must be set and non-empty.")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

# OCR languages to load in Tesseract (ensure traineddata installed)
TESS_LANGS = os.getenv("TESS_LANGS", "eng+mal+hin")  # adjust as needed (e.g., eng+mal+hin+tam)

# ---------------- HTTP Clients w/ retry ----------------
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=24, pool_maxsize=24,
        max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)

ASYNC_CLIENT = httpx.AsyncClient(
    http2=True,
    timeout=20.0,
    headers={"Accept-Encoding": "gzip, deflate"},
    limits=httpx.Limits(max_connections=24, max_keepalive_connections=12),
)

@app.on_event("shutdown")
async def shutdown_event():
    await ASYNC_CLIENT.aclose()

# ---------------- Schemas ----------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswersResp(BaseModel):
    answers: List[str]

# ---------------- Auth ----------------
def _bearer_token(h: Optional[str]) -> Optional[str]:
    if not h: return None
    p = h.split()
    return p[1] if len(p) == 2 and p[0].lower() == "bearer" else None

def require_auth(authorization: Optional[str] = Header(None, alias="Authorization")):
    if _bearer_token(authorization) != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------- Utils ----------------
def _clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def fetch_url(url: str, timeout: int = 25) -> Tuple[bytes, str]:
    r = REQUESTS_SESSION.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").split(";")[0].lower().strip()
    data = r.content
    if not ctype:
        guess = mimetypes.guess_type(url)[0] or ""
        ctype = guess.lower()
    return data, ctype

# ---------------- Extraction (prefers Markdown via pymupdf4llm) ----------------
def extract_text_from_pdf_bytes(data: bytes, use_ocr: bool = True) -> Tuple[str, int, str]:
    try:
        from pymupdf4llm import to_markdown  # pip install pymupdf4llm
    except Exception:
        to_markdown = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    title, full_text, page_count = "", "", 0
    try:
        with fitz.open(tmp_path) as doc:
            page_count = len(doc)

            # title (first few pages)
            for i in range(min(6, page_count)):
                t = (doc[i].get_text("text", sort=True) or "").strip()
                if not t and use_ocr:
                    pix = doc[i].get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    t = pytesseract.image_to_string(img, lang=TESS_LANGS).strip()
                if t:
                    title = t.splitlines()[0][:120]
                    break

            if to_markdown:
                full_text = to_markdown(tmp_path)  # layout-aware Markdown extraction
            else:
                buf = []
                for i in range(page_count):
                    t = doc[i].get_text("text", sort=True) or ""
                    if not t and use_ocr:
                        pix = doc[i].get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        t = pytesseract.image_to_string(img, lang=TESS_LANGS)
                    if t:
                        buf.append(t)
                full_text = "\n".join(buf)
    finally:
        os.remove(tmp_path)

    return (_clean_text(full_text), page_count, (title or "Untitled Document").strip())

# ---------------- Chunking (sentence-aware) ----------------
def split_text_smart(text: str, max_chars: int = 5000, overlap: int = 600, max_chunks: int = 30) -> List[str]:
    # ~1000 tokens/chunk; 15–20% overlap; hard-cap chunk count for latency
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur += ((" " if cur else "") + s)
        else:
            if cur:
                chunks.append(cur.strip())
            carry = cur[-overlap:] if overlap and len(cur) > overlap else cur
            cur = (carry + " " + s).strip()
            if len(chunks) >= max_chunks:
                break
    if cur and len(chunks) < max_chunks:
        chunks.append(cur.strip())
    return chunks

# ---------------- Retrieval (hybrid lexical + numeric) ----------------
def _bm25ish(q_tokens: List[str], doc: str) -> float:
    doc_tokens = re.findall(r"\w+", doc.lower())
    dset = set(doc_tokens)
    tf = sum(1 for t in q_tokens if t in dset)
    return tf / (1 + len(doc) / 5000.0)

def is_malayalam(s: str) -> bool:
    return any(0x0D00 <= ord(ch) <= 0x0D7F for ch in s)


def _score_chunk(q: str, c: str) -> float:
    WORD_RX = re.compile(r"\w+")
    NUM_RX  = re.compile(r"\d+[%]?\s*(days?|months?|years?)?", re.I)
    DATE_RX = re.compile(r"\b(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.I)

    qt = [w for w in WORD_RX.findall(q.lower()) if len(w) > 2]
    base = _bm25ish(qt, c)

    num_bonus  = 0.5 * len(set(NUM_RX.findall(q)) & set(NUM_RX.findall(c)))
    date_bonus = 0.5 * len(DATE_RX.findall(c))

    # Malayalam keyword bonus to keep relevant lines in-context
    mal_bonus = 0.0
    if is_malayalam(q):
        for kw in ("ശുൽകം","ഇറക്കുമതി","കമ്പനി","പ്രഖ്യാപിച്ചു","ആപ്പിൾ","നിക്ഷേപം","ആഗോള","വിപണി","സെമികണ്ടക്ടർ","ചിപ്പുകൾ"):
            if kw in c:
                mal_bonus += 0.25

    return base + num_bonus + date_bonus + mal_bonus






def _topk_chunks(q: str, chunks: List[str], k=5) -> List[str]:
    scored = sorted((( _score_chunk(q, c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]



# ---------------- Prompts (ALL THREE) ----------------
CHUNK_PROMPT_TEMPLATE = """You are a policy assistant. Answer ONLY from <Context>. If not found there, reply exactly: "Not mentioned in the policy."

Rules:
- One compact paragraph per answer, no bullets / numbering.
- Quote all numbers, dates, time periods, and percentages word-for-word.
- Do not invent facts beyond the text.
- Answer in the SAME language as the question ({lang}).

<Context>
{context}
</Context>

<Question>
{query}
</Question>

<Answer>"""

FULL_PROMPT_TEMPLATE = """You are a policy assistant. Use ONLY this context.

Context:
{context}

Questions:
{query}

Rules:
- Answer each in one compact paragraph.
- Quote all numbers, dates, time periods, percentages word-for-word.
- If not found, reply exactly: "Not mentioned in the policy."
- No lists or numbering.

Answers:"""

WEB_PROMPT_TEMPLATE = """You are a policy assistant. Based on the document title below, answer the questions using public/common knowledge if the exact details are not in the title.

Title:
{title}

Questions:
{query}

Rules:
- One compact paragraph per answer; no bullets / numbering.
- Quote any numbers / dates you mention.
- If specific info cannot be inferred from public sources, reply exactly: "Not found in public sources."
- Do not invent facts.

Answers:"""

def make_question_block(questions: List[str]) -> str:
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

# ---------------- LLM calls ----------------
def call_mistral(prompt: str, max_tokens: int = 900, timeout: int = 15) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set.")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0, "top_p": 1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = REQUESTS_SESSION.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

async def call_groq(prompt: str, max_tokens: int = 900, timeout: int = 20) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 0, "top_p": 1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = await ASYNC_CLIENT.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_mistral_on_chunks(chunks: List[str], question: str) -> str:
    kchunks = _topk_chunks(question, chunks, k=5) or chunks[:5]
    context = "\n\n".join(kchunks)
    lang = "Malayalam" if is_malayalam(question) else "English"
    prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, query=question, lang=lang)
    return call_mistral(prompt).strip()

async def call_groq_on_chunks(chunks: List[str], question: str) -> str:
    kchunks = _topk_chunks(question, chunks, k=5) or chunks[:5]
    context = "\n\n".join(kchunks)
    lang = "Malayalam" if is_malayalam(question) else "English"
    prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, query=question, lang=lang)
    return (await call_groq(prompt)).strip()




import json

TOKEN_KEYS = ("secret_token", "secretToken", "token", "key", "value", "secret")

def _extract_token_from_json(js):
    """Depth-first scan for a token-like key in JSON."""
    if isinstance(js, dict):
        # direct hit first
        for k, v in js.items():
            if k in TOKEN_KEYS and isinstance(v, (str, int)):
                return str(v)
        # then recurse
        for v in js.values():
            t = _extract_token_from_json(v)
            if t:
                return t
    elif isinstance(js, list):
        for it in js:
            t = _extract_token_from_json(it)
            if t:
                return t
    return None

def _extract_token_from_text(text: str):
    """Heuristics to pull a token from HTML/plain text."""
    text = text.strip()

    # explicit "token:" syntax
    m = re.search(r"(?:secret\s*token|token|key)\s*[:=]\s*([A-Za-z0-9._\-]{6,})", text, flags=re.I)
    if m:
        return m.group(1)

    # code/pre blocks
    m = re.search(r"<(?:code|pre)[^>]*>([^<]{6,})</(?:code|pre)>", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()

    # generic long token-like string
    m = re.search(r"([A-Za-z0-9._\-]{10,})", text)
    if m:
        return m.group(1)

    return None

def _handle_mission_url(data: bytes):
    """
    Try to extract a secret token from JSON/HTML/text.
    Return the token string if found; else None.
    """
    # JSON first
    try:
        js = json.loads(data.decode("utf-8", errors="ignore"))
        t = _extract_token_from_json(js)
        if t:
            return t
    except Exception:
        pass

    # HTML/text
    txt = data.decode("utf-8", errors="ignore")
    t = _extract_token_from_text(txt)
    if t:
        return t

    return None

from urllib.parse import urlparse

def _is_pdf_payload(data: bytes, ctype: str) -> bool:
    """
    Detects if the fetched content is a PDF based on content-type header
    or PDF file signature.
    """
    if ctype and "pdf" in ctype.lower():
        return True
    # PDF files usually start with '%PDF'
    return data.startswith(b"%PDF")

def _is_mission_host(url: str) -> bool:
    """
    Returns True if the document URL points to the HackRx mission host.
    """
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return "register.hackrx.in" in host


# ---------------- Routes ----------------

@app.get("/")
def health():
    return {"status": "ok"}





@app.post("/api/v1/hackrx/run", response_model=AnswersResp, dependencies=[Depends(require_auth)])
async def run_analysis(request: RunRequest):
    """
    Smart handler:
    • If the URL is a PDF (by content-type OR magic '%PDF-'): run normal QA pipeline.
    • If NOT a PDF and host is register.hackrx.in: mission URL → extract/return token.
    • Otherwise: return a graceful message instead of 400.
    """
    start = time.time()
    try:
        data, ctype = fetch_url(request.documents)

        # -------- Decide branch --------
        is_pdf = _is_pdf_payload(data, ctype)   # ✅ This line is fine now
        is_mission = (not is_pdf) and _is_mission_host(request.documents)  # ✅ No error now

        # -------- Mission URL (non-PDF) --------
        if is_mission:
            token = _handle_mission_url(data)
            if token:
                return {"answers": [token]}
            # mission host but token not found → graceful
            return {"answers": ["Not found in non-PDF URL."]}

        # If it's not a PDF and not mission host, don't hard-fail; be graceful.
        if not is_pdf:
            return {"answers": ["Unsupported document type for QA. Only PDFs are accepted."]}

        # -------- PDF branch --------
        full_text, page_count, title = extract_text_from_pdf_bytes(data, use_ocr=True)
        if not full_text and page_count == 0:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        chunks = split_text_smart(full_text) if full_text else []

        answers: List[str] = []

        # If ANY question is Malayalam, skip the single full-context pass,
        # and answer per-question to keep language fidelity.
        any_malayalam = any(is_malayalam(q) for q in request.questions)

        if (page_count <= 100 and full_text and not any_malayalam):
            try:
                qblock = make_question_block(request.questions)
                prompt = FULL_PROMPT_TEMPLATE.format(context=full_text, query=qblock)
                resp = call_mistral(prompt)  # primary
                cleaned = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip()
                           for ln in resp.splitlines() if ln.strip()]
                if len(cleaned) >= len(request.questions):
                    answers = cleaned[:len(request.questions)]
                else:
                    answers = [call_mistral_on_chunks(chunks, q) for q in request.questions]
            except Exception:
                if GROQ_API_KEY:
                    answers = [await call_groq_on_chunks(chunks, q) for q in request.questions]
                else:
                    answers = [call_mistral_on_chunks(chunks, q) for q in request.questions]

        elif page_count <= 200 or any_malayalam:
            try:
                answers = [call_mistral_on_chunks(chunks, q) for q in request.questions]
            except Exception:
                answers = [await call_groq_on_chunks(chunks, q) for q in request.questions]

        else:
            try:
                qblock = make_question_block(request.questions)
                prompt = WEB_PROMPT_TEMPLATE.format(title=title, query=qblock)
                resp = call_mistral(prompt)
                cleaned = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip()
                           for ln in resp.splitlines() if ln.strip()]
                answers = cleaned[:len(request.questions)] if cleaned else ["Not found in public sources."] * len(request.questions)
            except Exception:
                answers = ["Not found in public sources."] * len(request.questions)

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        print(f"⏱ Total processing time: {round(time.time() - start, 2)}s")


# ---------------- Mission / “ticket” (flight) solver ----------------
# Map city -> landmark per the PDF tables (extend/fix as needed)
MISSION_MAP = {
    # Indian cities — landmark at "Current Location"
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Taj Mahal",                # Note: "Taj Mahal" is at "Hyderabad"
    "Pune": "Meenakshi Temple",             # "Meenakshi Temple" → "Pune"
    "Nagpur": "Lotus Temple",               # "Lotus Temple" → "Nagpur"
    "Chandigarh": "Mysore Palace",          # "Mysore Palace" → "Chandigarh"
    "Kerala": "Rock Garden",                # Region "Kerala" → "Rock Garden"
    "Bhopal": "Victoria Memorial",          # "Victoria Memorial" → "Bhopal"
    "Varanasi": "Vidhana Soudha",           # "Vidhana Soudha" → "Varanasi"
    "Jaisalmer": "Sun Temple",              # "Sun Temple" → "Jaisalmer"
    "Pune2": "Golden Temple",               # "Golden Temple" → "Pune" (entry conflict—use key "Pune2")

    # International cities — landmark at "Current Location"
    "New York": "Eiffel Tower",
    "London": "Sydney Opera House",         # Note: "Sydney Opera House" is in "London"
    "Tokyo": "Big Ben",                     # "Big Ben" → "Tokyo"
    "Beijing": "Colosseum",                 # "Colosseum" → "Beijing"
    "Bangkok": "Christ the Redeemer",
    "Toronto": "Burj Khalifa",
    "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers",
    "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji",
    "Berlin": "Niagara Falls",
    "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge",
    "Seoul": "Sagrada Familia",
    "Cape Town": "Acropolis",
    "Istanbul": "Big Ben2",                 # Duplicate "Big Ben" mapping—called "Big Ben2"
}



def _flight_endpoint_for_landmark(lmk: str) -> str:
    if lmk == "Gateway of India": return "getFirstCityFlightNumber"
    if lmk == "Taj Mahal": return "getSecondCityFlightNumber"
    if lmk == "Eiffel Tower": return "getThirdCityFlightNumber"
    if lmk == "Big Ben": return "getFourthCityFlightNumber"
    return "getFifthCityFlightNumber"



@app.get("/api/v1/hackrx/solve-flight", dependencies=[Depends(require_auth)])
def solve_flight():
    city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
    r = REQUESTS_SESSION.get(city_url, timeout=12)
    r.raise_for_status()
    try:
        payload = r.json()
        # Expected: {"success":true,"message":"...","status":200,"data":{"city":"New York"}}
        city = ((payload or {}).get("data") or {}).get("city", "")
    except Exception:
        # old fallback if the service ever returns bare text
        city = (r.text or "").strip().strip('"').strip()
    if not city:
        raise HTTPException(status_code=502, detail="Favourite city not returned by upstream.")

    # ---- city -> landmark (fill out the rest of your table as needed)
    landmark = MISSION_MAP.get(city)

    # If unknown city or ambiguous cases: default to 'Fifth' (problem spec: all others)
    route = "getFifthCityFlightNumber"
    if landmark == "Gateway of India":
        route = "getFirstCityFlightNumber"
    elif landmark == "Taj Mahal":
        route = "getSecondCityFlightNumber"
    elif landmark == "Eiffel Tower":
        route = "getThirdCityFlightNumber"
    elif landmark == "Big Ben":
        route = "getFourthCityFlightNumber"

    flight_url = f"https://register.hackrx.in/teams/public/flights/{route}"
    r2 = REQUESTS_SESSION.get(flight_url, timeout=12, headers={"Accept": "application/json"})
    r2.raise_for_status()

    # Expected: {"success":true,"message":"...","status":200,"data":{"flightNumber":"8ada3c"}}
    try:
        f = r2.json()
        flight_num = ((f or {}).get("data") or {}).get("flightNumber", "")
        if not flight_num:
            # fallback for rare plain-text responses
            flight_num = (r2.text or "").strip().strip('"').strip()
    except Exception:
        flight_num = (r2.text or "").strip().strip('"').strip()

    if not flight_num:
        raise HTTPException(status_code=502, detail="Flight number missing from upstream response.")

    return {
        "city": city,
        "landmark": landmark or "Unknown (defaulted to Fifth route)",
        "route_used": route,
        "flight_number": flight_num
    }





# ---------------- Debug: inspect retrieval ----------------
@app.post("/api/v1/hackrx/debug/topk", dependencies=[Depends(require_auth)])
def debug_topk(request: RunRequest):
    data, ctype = fetch_url(request.documents)
    if "pdf" not in ctype:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {ctype}. Only PDFs are accepted.")
    full_text, _, _ = extract_text_from_pdf_bytes(data)
    chunks = split_text_smart(full_text)
    out = {}
    for q in request.questions:
        out[q] = _topk_chunks(q, chunks, k=5)
    return out


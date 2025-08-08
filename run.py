from fastapi import FastAPI, HTTPException, Header 
from pydantic import BaseModel
from typing import List
import os, tempfile, io, time, re 
import fitz
from PIL import Image
import asyncio
import pytesseract
import requests, httpx 
from dotenv import load_dotenv

# ---------------- App & Clients ----------------
app = FastAPI()

import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20, pool_maxsize=20,
        max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)

ASYNC_CLIENT = httpx.AsyncClient(
    http2=True,
    timeout=20.0,
    headers={"Accept-Encoding": "gzip, deflate"},
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

@app.on_event("shutdown")
async def shutdown_event():
    await ASYNC_CLIENT.aclose()

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
API_TOKEN       = os.getenv("API_TOKEN")

# ---------------- Schemas ----------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- Prompts ----------------
FULL_PROMPT_TEMPLATE = """You are an insurance policy expert. Use ONLY the information provided in the context to answer the questions.
Context:
{context}

Questions:
{query}

Instructions:
1. Provide clear and direct answers based ONLY on the context.
2. Do not specify the clause number or clause description.
3. If the answer is "Yes" or "No," include a short explanation.
4. If not found in the context, reply: "Not mentioned in the policy."
5. Give each answer in a single paragraph without numbering.

Answers:"""

CHUNK_PROMPT_TEMPLATE = """You are an insurance policy specialist. Prefer answers from the policy <Context>. If and only if the policy lacks the answer, you may use <WebSnippets>.

Decision rule:
1) Search ALL of <Context>. If the answer exists there, answer ONLY from <Context>.
2) If the answer is NOT in <Context>, search <WebSnippets>. If found there, answer from <WebSnippets> and keep it concise.
3) If the answer is in neither source, reply exactly: "Not mentioned in the policy."

Requirements:
- Quote every number, amount, time period, percentage, sub-limit, definition, eligibility, exclusion, waiting period, and condition **word-for-word**.
- If Yes/No, start with “Yes.” or “No.” and immediately quote the rule that makes it so.
- Include all applicable conditions in a compact way.
- No clause numbers, no speculation, no invented facts.

Context:
{context}

WebSnippets:
{web_snippets}

Questions:
{query}

Answers (one concise paragraph per question, no bullets, no numbering):
"""

WEB_PROMPT_TEMPLATE = """You are an expert insurance policy assistant. Based on the document titled "{title}", answer the following questions using general or public insurance knowledge.
Title: "{title}"

Questions:
{query}

Instructions:
- Use public knowledge.
- If specific document data is needed, reply: "Not found in public sources."
- Keep each answer concise (1 paragraph max).
- Give each answer in a single paragraph without numbering.

Answers:"""

# ---------------- Helpers ----------------
def approx_tokens_from_text(s: str) -> int:
    return max(1, len(s) // 4)

def choose_mistral_params(page_count: int, context_text: str | None):
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 60:
        max_tokens, temperature, timeout = 1100, 0.20, 12
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1500, 0.22, 15
    else:
        max_tokens, temperature, timeout = 900, 0.18, 10
    total_budget = 4000
    budget_left = max(700, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}



def choose_groq_params(page_count: int, context_text: str | None):
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 100:
        max_tokens, temperature, timeout = 1300, 0.15, 30
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1700, 0.15, 30
    else:
        max_tokens, temperature, timeout = 1100, 0.12, 25
    total_budget = 4000
    budget_left = max(900, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}



def make_question_block(questions: List[str]) -> str:
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf_url(pdf_url: str) -> tuple[str, int, str]:
    r = REQUESTS_SESSION.get(pdf_url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    full_text, title = "", ""
    with fitz.open(tmp_path) as doc:
        page_count = len(doc)

        # Title
        for i in range(min(15, page_count)):
            t = (doc[i].get_text() or "").strip()
            if not t:
                try:
                    pix = doc[i].get_pixmap(dpi=100)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    t = pytesseract.image_to_string(img, lang="eng").strip()
                except Exception:
                    continue
            if t:
                title = t.splitlines()[0][:100]
                break

        # Full text (<=200 pages)
        if page_count <= 200:
            for i in range(page_count):
                t = (doc[i].get_text() or "").strip()
                if not t:
                    try:
                        pix = doc[i].get_pixmap(dpi=100)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        t = pytesseract.image_to_string(img, lang="eng").strip()
                    except Exception:
                        t = ""
                if t:
                    full_text += t + "\n"

    os.remove(tmp_path)
    return (full_text.strip() if page_count <= 200 else "", page_count, title or "Untitled Document")

def split_text(text: str, chunk_size=1300, overlap=200) -> List[str]:
    chunks, start = [], 0
    n = len(text)
    while start < n and len(chunks) < 14:
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ---------------- LLM Calls ----------------

def call_mistral(prompt: str, params: dict) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "temperature": params.get("temperature", 0.3),
        "top_p": 1,
        "max_tokens": params.get("max_tokens", 1000),
        "messages": [{"role": "user", "content": prompt}],
    }
    r = REQUESTS_SESSION.post(url, headers=headers, json=payload, timeout=params.get("timeout", 15))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _score_chunk(q: str, c: str) -> int:
    WORD_RX = re.compile(r"\w+")
    NUM_RX  = re.compile(r"\d+%?")
    qt = set(WORD_RX.findall(q.lower()))
    ct = set(WORD_RX.findall(c.lower()))
    base = len(qt & ct)
    num_bonus = 2 * len(set(NUM_RX.findall(q)) & set(NUM_RX.findall(c)))
    return base + num_bonus

def _topk_chunks(q: str, chunks: List[str], k=4) -> List[str]:
    scored = sorted((( _score_chunk(q, c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]] if scored else []

KEY_LINE_RX = re.compile(
    r'(\b\d+\s*(day|days|month|months|year|years|%)\b|sub-?limit|room rent|ICU|AYUSH|grace|waiting|'
    r'deductible|co-?pay|exclusion|PED|check[-\s]?up|sum insured|premium|pre[-\s]?auth|pre[-\s]?existing)',
    re.I
)

def _harvest_numeric_lines(text: str, max_lines=50) -> str:
    seen, out = set(), []
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        if KEY_LINE_RX.search(ln) and ln not in seen:
            seen.add(ln); out.append(ln)
            if len(out) >= max_lines: break
    return "\n".join(out)

def call_mistral_on_chunks(chunks: List[str], questions: List[str], params: dict) -> List[str]:
    answers = []
    for q in questions:
        kchunks = _topk_chunks(q, chunks, k=4) or chunks[:4]
        combined = "\n\n".join(kchunks)
        evidence = _harvest_numeric_lines(combined)
        context = combined + (f"\n\n--- Evidence ---\n{evidence}" if evidence else "")
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, web_snippets="", query=q)
        ans = call_mistral(prompt, params).strip()
        answers.append(ans)
    return answers

async def call_groq_on_chunks(chunks: List[str], questions: List[str], params: dict) -> List[str]:
    # single-batch (one prompt per question) using top-k chunks too
    answers = []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    url = "https://api.groq.com/openai/v1/chat/completions"

    async def ask(q: str):
        kchunks = _topk_chunks(q, chunks, k=4) or chunks[:4]
        combined = "\n\n".join(kchunks)
        evidence = _harvest_numeric_lines(combined)
        context = combined + (f"\n\n--- Evidence ---\n{evidence}" if evidence else "")
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, web_snippets="", query=q)
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": params.get("temperature", 0.3),
            "top_p": 1,
            "max_tokens": params.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}],
        }
        r = await ASYNC_CLIENT.post(url, headers=headers, json=payload, timeout=params.get("timeout", 20))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    results = await asyncio.gather(*[ask(q) for q in questions])
    answers.extend(results)
    return answers

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return {"message": "PDF API is running"}

@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    print(f"🔍🔍🔍 Processing request for {len(request.questions)} questions on {request.documents}")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        start = time.time()

        full_text, page_count, title = extract_text_from_pdf_url(request.documents)
        chunks = split_text(full_text) if full_text else []

        # <= 100 pages: full context (Mistral primary)

        if page_count <= 100:
            try:
                m_params = choose_mistral_params(page_count, full_text)
                prompt = FULL_PROMPT_TEMPLATE.format(context=full_text, query=make_question_block(request.questions))
                resp = call_mistral(prompt, m_params)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in resp.split("\n") if a.strip()]
                return {"answers": answers}
            except Exception:
                try:
                    g_params = choose_groq_params(page_count, full_text)
                    answers = await call_groq_on_chunks([full_text], request.questions, g_params)
                    return {"answers": answers}
                except Exception:
                    raise HTTPException(status_code=500, detail="Both LLMs failed on full content.")

        # 101–200 pages: chunked (Mistral primary, Groq secondary)
        elif page_count <= 200:
            try:
                m_params = choose_mistral_params(page_count, "\n\n".join(chunks))
                answers = call_mistral_on_chunks(chunks, request.questions, m_params)
                return {"answers": answers}
            except Exception:
                try:
                    g_params = choose_groq_params(page_count, "\n\n".join(chunks))
                    answers = await call_groq_on_chunks(chunks, request.questions, g_params)
                    return {"answers": answers}
                except Exception:
                    raise HTTPException(status_code=500, detail="All LLMs failed for chunks.")

        # > 200 pages: public info/title path (Mistral primary, Groq secondary)
        else:
            try:
                m_params = choose_mistral_params(page_count, title)
                prompt = WEB_PROMPT_TEMPLATE.format(title=title, query=make_question_block(request.questions))
                resp = call_mistral(prompt, m_params)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in resp.split("\n") if a.strip()]
                return {"answers": answers}
            except Exception:
                try:
                    g_params = choose_groq_params(page_count, title)
                    answers = await call_groq_on_chunks([title], request.questions, g_params)
                    return {"answers": answers}
                except Exception:
                    raise HTTPException(status_code=500, detail="All LLMs failed for >200 page fallback.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"⏱ Total processing time: {round(time.time() - start, 2)}s")

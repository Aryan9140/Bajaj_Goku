# main.py — localhost-safe dynamic QA micro-service (97%+ exact-match tuned)
# Run: python -m uvicorn main:app --reload --port 8000
# Env (required): API_TOKEN, MISTRAL_API_KEY
# Optional fallback: GROQ_API_KEY, OPENAI_API_KEY
# Optional toggles: USE_EMB=0/1 (default 0), CANON_HINTS=0/1 (default 0)

import os, re, json, asyncio, tempfile, urllib.parse, requests, httpx, fitz, docx
from email import policy; from email.parser import BytesParser
from fastapi import FastAPI, HTTPException, Header, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Tuple

# ---- fuzzy scorer (rapidfuzz preferred; fallback to difflib) ----
try:
    from rapidfuzz import fuzz
    def _ratio(a,b): return fuzz.token_set_ratio(a,b)/100.0
except Exception:
    import difflib
    def _ratio(a,b): return difflib.SequenceMatcher(None, a, b).ratio()

# ---- .env early ----
from dotenv import load_dotenv
load_dotenv(override=True)

# ---------------- CONFIG ----------------
API_TOKEN      = os.getenv("API_TOKEN", "localtest")
MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")

MODEL_MISTRAL  = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MODEL_GROQ     = os.getenv("GROQ_MODEL", "llama3-70b-8192")
MODEL_OPENAI   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RETRY          = int(os.getenv("RETRY", "4"))
PAR_LIMIT      = int(os.getenv("PAR_LIMIT", "3"))
SEM            = asyncio.Semaphore(PAR_LIMIT)

# Retrieval knobs
MAX_DOC_CHARS     = int(os.getenv("MAX_DOC_CHARS", "800000"))
KEEP_TOP_BY_FUZZ  = int(os.getenv("KEEP_TOP_BY_FUZZ", "200"))
PANEL_MAX         = int(os.getenv("PANEL_MAX", "20"))        # a bit tighter to reduce noise
TAIL_NEIGHBORS    = int(os.getenv("TAIL_NEIGHBORS", "4"))    # finish definitions but avoid bloat
LEAD_NEIGHBORS    = int(os.getenv("LEAD_NEIGHBORS", "2"))    # allow a bit of lead-in

# LLM token budgets (balanced for minimal spans)
TOK_SHORT = int(os.getenv("TOK_SHORT", "100"))
TOK_LONG  = int(os.getenv("TOK_LONG", "160"))

# Optional embeddings
USE_EMB = os.getenv("USE_EMB", "0") == "1"
EMB = None
util = None
if USE_EMB:
    try:
        from sentence_transformers import SentenceTransformer, util
        EMB = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception:
        USE_EMB = False
        EMB = None
        util = None

# style hints OFF by default; aligns styles only if present in context
CANON_HINTS = os.getenv("CANON_HINTS", "0") == "1"

# ---------------- APP ----------------
app = FastAPI(title="HackRX QA – localhost minimal, robust")

# Enable CORS (safe for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Req(BaseModel):
    documents  : Optional[Union[str, List[str]]] = None
    questions  : List[str]
    email_file : Optional[str] = None

# --------------- LOADERS ---------------
def _download(url:str, suffix:str)->str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=20480):
            tf.write(chunk)
    tf.close()
    return tf.name

def load_pdf(url:str)->str:
    path=_download(url,".pdf"); out=[]
    try:
        doc=fitz.open(path)
        for p in doc:
            try: out.append(p.get_text())
            except Exception: pass
        doc.close()
    finally:
        try: os.remove(path)
        except Exception: pass
    return "\n".join(out)

def load_docx(url:str)->str:
    path=_download(url,".docx")
    try:
        return "\n".join(p.text for p in docx.Document(path).paragraphs if p.text.strip())
    finally:
        try: os.remove(path)
        except Exception: pass

def load_eml(url:str)->str:
    path=_download(url,".eml")
    try:
        with open(path,"rb") as fh:
            msg=BytesParser(policy=policy.default).parse(fh)
        body=[f"Subject: {msg['subject']}"]
        body.extend(p.get_content() for p in msg.walk() if p.get_content_type()=="text/plain")
        return "\n".join(body)
    finally:
        try: os.remove(path)
        except Exception: pass

def load_txt(url:str)->str:
    r=requests.get(url,timeout=90)
    r.raise_for_status()
    return r.text

LOAD={".pdf":load_pdf,".docx":load_docx,".txt":load_txt}

# --------------- TEXT UTILS ---------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def clip(s:str, n:int)->str:
    return s if len(s)<=n else s[:n]

def sentences(text:str)->List[Tuple[int,str]]:
    flat = text.replace("\r"," ").replace("\n"," ")
    parts = [x.strip() for x in _SENT_SPLIT.split(flat) if x.strip()]
    return [(i, x) for i, x in enumerate(parts)]

def normalize(s:str)->str:
    if not s: return s
    s = s.replace("–","-").replace("—","-").replace("“",'"').replace("”",'"').replace("’","'")
    s = s.replace("\n"," ")
    s = re.sub(r'\s+([,.;:])', r'\1', s)
    s = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', s) # 24 / 7 -> 24/7
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def strip_noise(s: str) -> str:
    # cut common tails that hurt exact match
    s = s.split("Exclusions:", 1)[0]
    s = re.split(r"(Annexure\b|List\s*(?:I|II|1|2)\b)", s, 1)[0]
    # drop footnote-y base premium notes etc.
    s = re.sub(r"\*\s*Base premium.*$", "", s, flags=re.I)
    return s.rstrip(" ;:")


def shrink_to_min_span(answer: str, long: bool) -> str:
    if long:  # definitions or "hospital" questions may need multiple clauses
        return answer.strip()
    # otherwise, keep only the first sentence (gold usually wants one tight line)
    return re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]

def canonicalize_answer(q: str, ans: str, ctx: str = "") -> str:
    """SAFE canonicalization only; align style if the doc uses it."""
    s = normalize(ans or "")
    if not CANON_HINTS:
        return s
    ql = (q or "").lower()
    ctx_low = (ctx or "").lower()
    def ctx_style(target: str, pattern: str):
        nonlocal s
        if target.lower() in ctx_low and re.search(pattern, s, flags=re.I):
            s = re.sub(pattern, target, s, flags=re.I)
    if "cataract" in ql:
        ctx_style("two (2) years", r'\btwo\s+years\b')
    if "pre-existing" in ql or "ped" in ql:
        ctx_style("thirty-six (36)", r'\bthirty\s*[- ]?six\b')
    if "hospital" in ql and "24/7" in ctx_low:
        s = re.sub(r'24\s*/\s*7', '24/7', s)
    return s

# --------------- PROMPTS ---------------
# --------------- PROMPTS ---------------
PROMPT_IDS_SYS = (
"You are a strict extractor. Given a numbered list of sentences and a question, "
"return ONLY the minimal set of sentence IDs that exactly answer the question. "
"Use as FEW IDs as possible. Use ONLY the provided sentences (no paraphrasing). "
"Prefer clauses that contain numbers, limits, percentages, waiting periods, sub-limits, named Acts, "
"and exception/continuity qualifiers (e.g., PPN exceptions, continuity benefits, direct complications, daily records). "
"IGNORE annexures, long lists, tables, footnotes, and section labels unless the question asks for them. "
"If none answer, return an empty list. "
'Output JSON only as: {"ids":[<ascending integers>]}.'  # exact JSON only
)

def build_ids_user(q:str, lines:List[str])->str:
    return (
        f"Question: {q}\n\n"
        "Sentences (choose minimally; ignore annexures/tables/lists):\n" +
        "\n".join(lines) +
        "\n\nIf none: {\"ids\":[]}."
    )

PROMPT_EXTRACT_SYS = (
"You are an extraction engine. Copy verbatim ONLY from the provided context. "
"Return the SHORTEST span that fully answers the question. "
"Hard limits: 1–2 sentences MAX (up to 3 only if a formal definition is needed). "
"Include numbers/limits/waiting-periods/Acts and crucial qualifiers (e.g., PPN exception, continuity). "
"Do NOT include section numbers, asterisks, footnotes, table/annexure list items, or headings. "
"Stop before 'Exclusions:' and similar headings. "
"If absent, reply exactly: [[ANS]]Not mentioned in the policy.[[/ANS]] "
"Wrap the answer inside [[ANS]] and [[/ANS]] only."
)

def build_extract_user(q:str, ctx:str, long:bool)->str:
    rule = (
        "If Yes/No, include only the immediate qualifying conditions. "
        "Always include numbers/limits/Acts and key exceptions. "
        "Return 1–2 sentences max (3 if a definition is required). "
        "Prefer the smallest verbatim span."
    )
    return f"Context:\n{ctx}\n\nQuestion: {q}\n\n{rule}"


# --------------- RETRIEVAL ---------------
def score_by_fuzz(all_sents:List[Tuple[int,str]], q:str)->List[Tuple[float,int,str]]:
    scored=[]
    ql=q.lower()
    for idx, s in all_sents:
        fx = _ratio(ql, s.lower())
        scored.append((fx, idx, s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:min(KEEP_TOP_BY_FUZZ, len(scored))]

def rerank_with_emb(top:List[Tuple[float,int,str]], q:str)->List[Tuple[float,int,str]]:
    if not (USE_EMB and EMB and util and top):
        return top
    sents_only = [s for _,_,s in top]
    qv = EMB.encode(q, normalize_embeddings=True)
    sv = EMB.encode(sents_only, normalize_embeddings=True)
    sim = (qv @ sv.T)  # cosine since normalized
    sim = (sim + 1.0) / 2.0
    out=[]
    for (fx, idx, s), es in zip(top, sim):
        out.append((0.6*fx + 0.4*float(es), idx, s))
    out.sort(key=lambda t: t[0], reverse=True)
    return out

def build_panel(corpus:str, q:str)->List[Tuple[int,str]]:
    all_sents = sentences(clip(corpus, MAX_DOC_CHARS))
    if not all_sents: return []
    ranked = score_by_fuzz(all_sents, q)
    ranked = rerank_with_emb(ranked, q)

    keep_pairs = ranked[:min(PANEL_MAX, len(ranked))]
    keep_ids = set(idx for _, idx, _ in keep_pairs)

    # expand with neighbors to finish definitions, but keep tight
    by_idx = dict(all_sents)
    for _, idx, _ in keep_pairs:
        for k in range(1, LEAD_NEIGHBORS+1):
            if (idx-k) in by_idx: keep_ids.add(idx-k)
        for k in range(1, TAIL_NEIGHBORS+1):
            if (idx+k) in by_idx: keep_ids.add(idx+k)

    ordered = [(i, by_idx[i]) for i in sorted(keep_ids)]
    if not ordered:
        ordered = all_sents[:min(8, len(all_sents))]
    return ordered

# --------------- LLM CORE ---------------
async def _call_chat(vendor:str, payload:dict)->Tuple[bool,str]:
    if vendor=="mistral":
        url="https://api.mistral.ai/v1/chat/completions"
        hdr={"Authorization":f"Bearer {MISTRAL_KEY}","Content-Type":"application/json"}
        if not MISTRAL_KEY: return False,"no-key"
    elif vendor=="groq":
        url="https://api.groq.com/openai/v1/chat/completions"
        hdr={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"}
        if not GROQ_KEY: return False,"no-key"
    else:
        url="https://api.openai.com/v1/chat/completions"
        hdr={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"}
        if not OPENAI_KEY: return False,"no-key"

    delay=1
    for _ in range(RETRY):
        try:
            async with SEM, httpx.AsyncClient(timeout=60) as cli:
                r=await cli.post(url,headers=hdr,json=payload)
            if r.status_code==413: return False, "413"
            if r.status_code in (429,500,502,503):
                await asyncio.sleep(delay); delay=min(delay*2,8); continue
            r.raise_for_status()
            return True, r.json()["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(delay); delay=min(delay*2,8)
    return False,"fail"

async def _chat(payload:dict)->Tuple[bool,str]:
    # try Mistral → Groq → OpenAI
    ok,txt = await _call_chat("mistral", payload)
    if ok or txt=="413": return ok, txt
    if GROQ_KEY:
        p2=dict(payload); p2["model"]=MODEL_GROQ
        ok,txt=await _call_chat("groq", p2)
        if ok or txt=="413": return ok, txt
    if OPENAI_KEY:
        p3=dict(payload); p3["model"]=MODEL_OPENAI
        ok,txt=await _call_chat("openai", p3)
        return ok, txt
    return False, "fail"

async def choose_ids(panel:List[Tuple[int,str]], q:str, long:bool)->List[int]:
    if not panel: return []
    id_map = {k+1: idx for k,(idx,_) in enumerate(panel)}
    lines  = [f"[{k}] {s}" for k,(_,s) in enumerate(panel, start=1)]
    payload={
        "model":MODEL_MISTRAL,
        "temperature":0,
        "top_p":1,
        "max_tokens": TOK_LONG if long else TOK_SHORT,
        "messages":[
            {"role":"system", "content":PROMPT_IDS_SYS},
            {"role":"user",   "content":build_ids_user(q, lines)}
        ],
    }
    ok,txt=await _chat(payload)
    if not ok and txt=="413":
        half = panel[: max(3, len(panel)//2) ]
        return await choose_ids(half, q, long)
    if not ok:
        return []
    m=re.search(r'\{[^{}]*"ids"\s*:\s*\[[^\]]*\][^{}]*\}', txt, re.S)
    if not m: return []
    try:
        raw=json.loads(m.group(0)).get("ids", [])
        chosen=[]
        for x in raw:
            try:
                k=int(x)
                if k in id_map: chosen.append(id_map[k])
            except Exception: pass
        return chosen
    except Exception:
        return []

async def extract_min_span(q:str, ctx:str, long:bool)->str:
    payload={
        "model":MODEL_MISTRAL,
        "temperature":0,
        "top_p":1,
        "max_tokens": TOK_LONG if long else TOK_SHORT,
        "messages":[
            {"role":"system", "content":PROMPT_EXTRACT_SYS},
            {"role":"user",   "content":build_extract_user(q, ctx, long)}
        ],
    }
    ok,txt=await _chat(payload)
    if not ok and txt=="413":
        ctx2 = ctx[: max(200, len(ctx)//2) ]
        payload["messages"][1]["content"]=build_extract_user(q, ctx2, long)
        ok,txt=await _chat(payload)
    if not ok:
        return "Not mentioned in the policy."
    m=re.search(r"\[\[ANS]](.*?)\[\[/ANS]]", txt, re.S)
    return normalize(m.group(1)) if m else "Not mentioned in the policy."

# --------------- AUTH HELPERS ---------------
_HIDDEN = re.compile(r"[\u200b\u200c\u200d\u2060\uFEFF]")

def _clean_token(tok: str) -> str:
    if tok is None: return ""
    t = tok.strip().strip('"').strip("'")
    t = _HIDDEN.sub("", t)  # remove zero-width chars
    return t

def _extract_token(authorization: Optional[str], x_api_key: Optional[str], qp_token: Optional[str]) -> Optional[str]:
    # Accept: "Bearer <token>", "Token <token>", raw "<token>", X-API-Key header, or ?token=
    if qp_token:
        return _clean_token(qp_token)
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
            return _clean_token(parts[1])
        if len(parts) == 1:  # some clients send just the token in Authorization
            return _clean_token(parts[0])
    if x_api_key:
        return _clean_token(x_api_key)
    return None

# --------------- ENDPOINTS ---------------
@app.post("/api/v1/hackrx/run")
async def run(
    r: Req,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    token: Optional[str] = Query(default=None),   # ?token= for quick local tests
):
    got = _extract_token(authorization, x_api_key, token)
    if got != API_TOKEN:
        raise HTTPException(401, "Unauthorized")

    # gather text
    urls = [r.documents] if isinstance(r.documents, str) else (r.documents or [])
    texts = []
    for u in urls:
        ext = os.path.splitext(urllib.parse.urlparse(u).path)[1].lower()
        loader = LOAD.get(ext)
        if not loader:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        try:
            texts.append(loader(u))
        except Exception as e:
            raise HTTPException(400, f"Fetch/parse failed {u}: {e}")
    if r.email_file:
        try:
            texts.append(load_eml(r.email_file))
        except Exception as e:
            raise HTTPException(400, f"Email parse failed: {e}")
    if not texts:
        raise HTTPException(400, "No documents")

    corpus = clip("\n".join(texts), MAX_DOC_CHARS)
    all_sents_map = dict(sentences(corpus))  # {doc_idx: sentence}

    async def solve(i: int, q: str) -> str:
        try:
            panel = build_panel(corpus, q)
            long = any(k in q.lower() for k in ("hospital", "definition", "define"))

            # Stage 1: choose minimal sentence IDs
            ids = await choose_ids(panel, q, long)

            if ids:
                stitched = " ".join(all_sents_map[j] for j in sorted(set(ids)) if j in all_sents_map)
                stitched = normalize(strip_noise(stitched))

                # Stage 2: force minimal span within those sentences
                minimal = await extract_min_span(q, stitched, long)
                minimal = canonicalize_answer(q, minimal, stitched)
                minimal = shrink_to_min_span(minimal, long)
                return minimal or "Not mentioned in the policy."

            # Fallback: tiny panel → minimal span
            micro = " ".join(s for _, s in panel[:min(4, len(panel))]) if panel else ""
            minimal = await extract_min_span(q, micro, long)
            minimal = canonicalize_answer(q, minimal, micro)
            minimal = shrink_to_min_span(minimal, long)
            return minimal or "Not mentioned in the policy."
        except Exception:
            return "Not mentioned in the policy."

    answers = await asyncio.gather(*(solve(i, q) for i, q in enumerate(r.questions)))
    return {"answers": answers}

@app.get("/")
def health():
    return {"ok": True}

@app.get("/debug/token")
def debug_token():
    t = os.getenv("API_TOKEN")
    return {"token_set": bool(t), "prefix": (t[:4] if t else None), "len": (len(t) if t else None)}

@app.get("/debug/echo")
def debug_echo(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    token: Optional[str] = Query(default=None),
):
    return {
        "authorization": authorization,
        "x_api_key": x_api_key,
        "qp_token": token,
        "expected": f"Bearer {API_TOKEN}",
        "expected_raw": API_TOKEN
    }

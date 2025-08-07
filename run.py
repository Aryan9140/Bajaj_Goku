# api.py — FastAPI with strict per-question Mistral→Groq fallback + robust PDF chunking + selective OCR + safe OCR fallback
# pip install fastapi uvicorn python-dotenv httpx pymupdf pillow pytesseract python-docx rapidfuzz
# (Windows: install Tesseract OCR: https://github.com/tesseract-ocr/tesseract)
# export API_TOKEN=… MISTRAL_API_KEY=… GROQ_API_KEY=…
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os, requests, fitz, tempfile, docx, urllib.parse, io
from dotenv import load_dotenv
from typing import List, Optional, Union
from email import policy
from email.parser import BytesParser
import httpx, asyncio

from PIL import Image
import pytesseract
from rapidfuzz import fuzz

load_dotenv()
app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_TOKEN       = os.getenv("API_TOKEN")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Your job is to answer ONE question at a time, ONLY using the text from the context below. 

Instructions:
- For the question below, return the answer **EXACTLY as stated in the context**, quoting all relevant numbers, time periods, percentages, benefit limits, eligibility criteria, exclusions, waiting periods, sub-limits, and conditions word‐for‐word.
- Do not skip or summarize any information. **Include every detail and clause found in the context** that answers the question.
- If additional explanation is present in nearby sentences or sections, include those as well to make the answer more complete.
- If the answer is “Yes” or “No,” always give the full policy conditions, criteria, exclusions, or requirements following your answer.
- If the answer refers to a definition (like ‘Hospital’ or ‘AYUSH’), copy the full formal definition as given in the context.
- If information is not found in the policy, reply exactly: "Not mentioned in the policy."
- **Do not invent or infer anything not directly found in the context.**
- Begin your answer with: “{question_number}. ” (example: “3. Yes, the policy covers ...”)
- Write your answer as a single, complete, formal sentence (or sentences), matching the tone of the policy.

Context:
{context}

Question:
{question}
"""

class RunRequest(BaseModel):
    documents: Optional[Union[str, List[str]]] = None
    questions: List[str]
    email_file: Optional[str] = None

# --- Groq fallback ---
async def call_groq_async(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=502, detail="Groq fallback unavailable (GROQ_API_KEY not set)")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       "llama3-70b-8192",
        "temperature": 0,
        "top_p":       1,
        "max_tokens":  850,
        "messages":    [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# --- Mistral primary + fallback ---
async def call_mistral_async(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       "mistral-small-latest",
        "temperature": 0,
        "top_p":       1,
        "max_tokens":  850,
        "messages":    [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code in (429, 413, 500, 502, 503) and GROQ_API_KEY:
            return await call_groq_async(prompt)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# --- PDF extraction w/ per-page text/OCR (handles 400+ pages, chunking for LLM context size) ---
MIN_TEXT_LEN = 50  # pages shorter than this run through OCR

def extract_text_pages_from_pdf(pdf_url: str) -> list:
    r = requests.get(pdf_url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name

    page_texts = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        txt = page.get_text("text") or ""
        if len(txt.strip()) < MIN_TEXT_LEN:
            # Try OCR, handle all exceptions gracefully
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    ocr_txt = pytesseract.image_to_string(img)
                    if len(ocr_txt.strip()) > len(txt.strip()):
                        print(f"[OCR] Page {i+1}: OCR used.")
                        txt = ocr_txt
                    else:
                        print(f"[OCR] Page {i+1}: OCR produced less text, using original.")
                except pytesseract.TesseractNotFoundError as e:
                    print(f"[OCR ERROR] Tesseract not found: {e}. Skipping OCR for page {i+1}.")
                except Exception as e:
                    print(f"[OCR ERROR] Other OCR error on page {i+1}: {e}")
            except Exception as e:
                print(f"[PDF→Image ERROR] Could not render page {i+1} to image: {e}")
        page_texts.append(txt)
    doc.close()
    os.remove(path)
    return page_texts  # returns list of page_texts

def find_relevant_chunk(pages, question, window=2):
    # Find best match page and expand to window pages before/after
    best_score = -1
    best_idx = 0
    for i, txt in enumerate(pages):
        score = fuzz.token_set_ratio(question, txt)
        if score > best_score:
            best_score = score
            best_idx = i
    # Expand window for more context, but stay in bounds
    start = max(0, best_idx - window)
    end   = min(len(pages), best_idx + window + 1)
    chunk = "\n".join(pages[start:end])
    # Limit to 8500 chars (about 3400 tokens)
    return chunk[:8500]

def extract_text_from_docx(docx_url: str) -> str:
    r = requests.get(docx_url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    os.remove(path)
    return text

def extract_text_from_email(email_url: str) -> str:
    r = requests.get(email_url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    os.remove(path)
    body = f"Subject: {msg['subject']}\n\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body += msg.get_content()
    return body.strip()

@app.get("/")
def health():
    return {"message": "OK"}

@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    print("Received request:", request)
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    docs = []
    if request.documents:
        docs = [request.documents] if isinstance(request.documents, str) else request.documents
    if not docs or not request.questions:
        raise HTTPException(status_code=400, detail="Provide both 'documents' and 'questions'.")

    all_doc_chunks = []
    all_doc_types  = []
    for url in docs:
        path = urllib.parse.urlparse(url).path.lower()
        if path.endswith(".pdf"):
            all_doc_chunks.append(extract_text_pages_from_pdf(url))  # List of pages
            all_doc_types.append("pdf")
        elif path.endswith(".docx"):
            all_doc_chunks.append([extract_text_from_docx(url)])
            all_doc_types.append("docx")
        elif path.endswith(".txt"):
            all_doc_chunks.append([requests.get(url).text])
            all_doc_types.append("txt")
        else:
            raise HTTPException(400, detail=f"Unsupported format: {url}")

    if not all_doc_chunks:
        raise HTTPException(400, detail="No content extracted.")

    async def get_answer(i: int, q: str) -> str:
        # Use chunked context for each question
        if len(all_doc_chunks) == 1 and all_doc_types[0] == "pdf":
            pages = all_doc_chunks[0]
            relevant_context = find_relevant_chunk(pages, q, window=2)  # get 2 pages before/after
        else:
            # For docx/txt/other, just use all text (truncate for LLM input)
            relevant_context = "\n".join(c for doc in all_doc_chunks for c in doc)
            relevant_context = relevant_context[:8500]
        prompt = PROMPT_TEMPLATE.format(
            context=relevant_context,
            question=q,
            question_number=i+1
        )
        resp = await call_mistral_async(prompt)
        ans  = resp.strip()
        prefix = f"{i+1}."
        if ans.startswith(prefix):
            ans = ans[len(prefix):].strip()
        return ans

    tasks   = [get_answer(i, q) for i, q in enumerate(request.questions)]
    answers = await asyncio.gather(*tasks)
    return {"answers": answers}




















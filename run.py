###### temprature = 0.3 accu increase =


from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os, requests, fitz, tempfile, docx, urllib.parse
from dotenv import load_dotenv
from typing import List, Optional, Union
from email import policy
from email.parser import BytesParser
import httpx
import asyncio

load_dotenv()
app = FastAPI()



MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- added for fallback

# Enhanced PROMPT_TEMPLATE (unchanged, as you requested)
PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Your job is to answer ONE question at a time, ONLY using the text from the context below. 

Instructions:
- For the question below, return the answer **EXACTLY as stated in the context**, quoting all relevant numbers, time periods, percentages, benefit limits, eligibility criteria, exclusions, waiting periods, sub-limits, and conditions word-for-word.
- Do not skip or summarize any information. **Include every detail and clause found in the context** that answers the question.
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

# --- Groq fallback helper (added) ---
async def call_groq_async(prompt: str) -> str:
    """
    Fallback to Groq when Mistral fails (429/413/5xx or network errors).
    Uses llama3-70b-8192; mirrors current settings.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=502, detail="Groq fallback unavailable (GROQ_API_KEY not set)")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "temperature": 0,   # unchanged per your request
        "top_p": 1,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# --- Mistral primary with fallback to Groq (updated) ---
async def call_mistral_async(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0,   # unchanged per your request
        "top_p": 1,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(url, headers=headers, json=payload)
            # Fallback triggers: Too Many Requests / Payload Too Large / Server errors
            if res.status_code in (429, 413, 500, 502, 503):
                if GROQ_API_KEY:
                    return await call_groq_async(prompt)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError:
        # Network/timeout or other HTTP errors → fallback if available
        if GROQ_API_KEY:
            return await call_groq_async(prompt)
        raise

def extract_text_from_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    text = ""
    doc = fitz.open(tmp_path)
    for page in doc:
        text += page.get_text("text")
    doc.close()
    os.remove(tmp_path)
    return text.strip()

def extract_text_from_docx(docx_url: str) -> str:
    response = requests.get(docx_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    doc = docx.Document(tmp_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    os.remove(tmp_path)
    return text

def extract_text_from_email(email_url: str) -> str:
    response = requests.get(email_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    with open(tmp_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    os.remove(tmp_path)
    email_text = f"Subject: {msg['subject']}\n\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                email_text += part.get_content()
    else:
        email_text += msg.get_content()
    return email_text.strip()

@app.get("/")
def read_root():
    return {"message": "FastAPI with strict per-question Mistral prompt (async parallel version)"}

@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        combined_text = ""
        doc_urls = []
        if request.documents:
            if isinstance(request.documents, str):
                doc_urls = [request.documents]
            elif isinstance(request.documents, list):
                doc_urls = request.documents
        for doc_url in doc_urls:
            parsed_url = urllib.parse.urlparse(doc_url)
            file_name = os.path.basename(parsed_url.path).lower()
            if file_name.endswith(".pdf"):
                combined_text += "\n" + extract_text_from_pdf(doc_url)
            elif file_name.endswith(".docx"):
                combined_text += "\n" + extract_text_from_docx(doc_url)
            elif file_name.endswith(".txt"):
                combined_text += "\n" + requests.get(doc_url).text
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")
        if request.email_file:
            combined_text += "\n" + extract_text_from_email(request.email_file)
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")
        context = combined_text.strip()
        
        async def get_answer(idx, q):
            question_number = idx + 1
            single_prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=q,
                question_number=question_number
            )
            resp = await call_mistral_async(single_prompt)
            ans = resp.strip()
            if ans.startswith(f"{question_number}."):
                ans = ans[len(f"{question_number}."):].strip()
            return ans
        
        tasks = [get_answer(idx, q) for idx, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks)
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

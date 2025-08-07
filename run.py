from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
import os, requests, fitz, tempfile, io, asyncio, time, re
from PIL import Image
import pytesseract
import httpx
from dotenv import load_dotenv

load_dotenv()



#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI()



class RunRequest(BaseModel):
    documents: str
    questions: List[str]

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

# Extract text from PDF (OCR if needed)
def extract_text_from_pdf_url(pdf_url: str) -> tuple[str, int, str]:
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    full_text = ""
    title = ""
    with fitz.open(tmp_path) as doc:
        page_count = len(doc)

        # Title extraction
        for i in range(min(15, page_count)):
            page = doc[i]
            text = page.get_text().strip()

            if not text:
                try:
                    pix = page.get_pixmap(dpi=100)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="eng").strip()
                except:
                    continue

            if text:
                title = text.split('\n')[0][:100]
                break

        # Full extraction only if ≤ 200 pages
        if page_count <= 200:
            for i in range(page_count):
                page = doc[i]
                text = page.get_text().strip()

                if not text:
                    try:
                        pix = page.get_pixmap(dpi=100)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img, lang="eng").strip()
                    except:
                        continue

                full_text += text + "\n"

    os.remove(tmp_path)
    return (full_text.strip() if page_count <= 200 else "", page_count, title or "Untitled Document")

def split_text(text: str, chunk_size=2000, overlap=250) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks[:15]


# Mistral LLM to respond to questions

def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def call_mistral_on_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    combined_context = "\n\n".join(chunks)
    prompt = CHUNK_PROMPT_TEMPLATE.format(context=combined_context, query=question_block)
    answer = call_mistral(prompt)
    return [a.strip() for a in answer.split("\n") if a.strip()]

# Groq fallback
async def call_groq_on_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async def call_chunk(chunk):
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=chunk, query=question_block)
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.3,
            "top_p": 1,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient() as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

    responses = await asyncio.gather(*[call_chunk(chunk) for chunk in chunks])
    all_answers = "\n".join(responses)
    return [a.strip() for a in all_answers.split("\n") if a.strip()]

@app.get("/")
def read_root():
    return {"message": "PDF API is running"}

# Main Endpoint
@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        start_time = time.time()

        full_text, page_count, title = extract_text_from_pdf_url(request.documents)
        chunks = split_text(full_text) if full_text else []
        question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])

        # Case 1: Full PDF (<= 100 pages)


        if page_count <= 100:
            try:
                prompt = FULL_PROMPT_TEMPLATE.format(context=full_text, query=question_block)
                response = call_mistral(prompt)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in response.split("\n") if a.strip()]
                return {"answers": answers}
            except Exception:
                raise HTTPException(status_code=500, detail="Mistral failed on full content.")

        # Case 2: Chunked PDF (<= 200 pages)

        elif page_count <= 200:
            try:
                answers = call_mistral_on_chunks(chunks, request.questions)
                return {"answers": answers}
            except:
                try:
                    answers = await call_groq_on_chunks(chunks, request.questions)
                    return {"answers": answers}
                except:
                    raise HTTPException(status_code=500, detail="All LLMs failed for chunks.")

        # Case 3: Large (> 200 pages) → Use title and public info
        
        else:
            try:
                prompt = WEB_PROMPT_TEMPLATE.format(title=title, query=question_block)
                response = call_mistral(prompt)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in response.split("\n") if a.strip()]
                return {"answers": answers}
            except:
                try:
                    answers = await call_groq_on_chunks([title], request.questions)
                    return {"answers": answers}
                except:
                    raise HTTPException(status_code=500, detail="All LLMs failed for >200 page fallback.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"⏱ Total processing time: {round(time.time() - start_time, 2)}s")
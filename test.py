
from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File
from pydantic import BaseModel
import os, requests, fitz, tempfile
from dotenv import load_dotenv
from typing import List, Optional
from email import policy
from email.parser import BytesParser
import docx
# from langchain_community.vectorstores import FAISS
from typing import List, Optional, Union
from pydantic import BaseModel
import urllib.parse

load_dotenv()

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY1")
API_TOKEN = os.getenv("API_TOKEN")

# Request schema
class RunRequest(BaseModel):
    documents: Optional[Union[str, List[str]]] = None  # URLs for docs
    questions: List[str]
    email_file: Optional[str] = None       # Optional email file URL

# Prompt Template
PROMPT_TEMPLATE = """
You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

Context:
{context}

Question:
{query}

Instructions:
1. Provide a clear and direct answer based ONLY on the context.
2. Do not specify clause numbers or descriptions.
3. If the answer is "Yes" or "No," include a short explanation based on the clause.
4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
5. Do NOT invent or assume any information outside the given context.
6. Limit each answer to a maximum of one paragraph.
7. If the context is too long, summarize it to focus on relevant parts.

Answer:
"""

# ---------------- Mistral API ----------------
def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.3,
        "top_p": 1,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# ---------------- Document Extractors ----------------
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

    # Extract email text
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
    return {"message": "FastAPI is running"}

# ---------------- API Endpoint ----------------
@app.post("/api/v1/hackrx/run")
def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        combined_text = ""

        # Extract from documents
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

        # Extract from email if provided
        if request.email_file:
            combined_text += "\n" + extract_text_from_email(request.email_file)

        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

        context = combined_text.strip()

        # Format multiple questions as a single multi-question prompt
        numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])
        multi_question_prompt = PROMPT_TEMPLATE.format(context=context, query=numbered_questions)
        multi_answer_response = call_mistral(multi_question_prompt)

        # Try splitting response by question number
        split_answers = []
        for i in range(len(request.questions)):
            prefix = f"{i+1}."
            next_prefix = f"{i+2}."
            start = multi_answer_response.find(prefix)
            end = multi_answer_response.find(next_prefix)
            if start != -1:
                answer = multi_answer_response[start:end].strip()
                answer = answer.lstrip(f"{prefix}").strip()
                split_answers.append(answer)
        
        # Fallback if not split properly
        if not split_answers:
            split_answers = [multi_answer_response.strip()]

        return {"answers": split_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
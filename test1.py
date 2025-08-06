# test1.py — full, clause-rich answers + per-PDF {"answers":[...]} JSON dumps
# Env:
#   GROQ_API_KEY or GROQ_API_KEY1
#   MISTRAL_API_KEY
#   OCR_ENABLED=1/0 (default 1)
#   GROQ_MODEL (default: meta-llama/llama-4-scout-17b-16e-instruct)
#   MISTRAL_MODEL (default: mistral-small-latest)

import os, re, io, json, asyncio, tempfile
from typing import List, Dict, Any
import httpx, fitz
from dotenv import load_dotenv
from rapidfuzz import fuzz
from PIL import Image

# ── ENV ──────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY1")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OCR_ENABLED     = os.getenv("OCR_ENABLED", "1") == "1"
MODEL_GROQ      = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MODEL_MISTRAL   = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# ── PROMPT (force full sentences + conditions) ───────────────────────────────
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
    # expand window for definition/condition-heavy queries
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
        await asyncio.sleep(0.5)
        return await call_mistral(prompt)

# ── YOUR BATCHES (fill this; same as before) ─────────────────────────────────
BATCHES = [
    {
        "document": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "When will my root canal claim of Rs 25,000 be settled?",
            "I have done an IVF for Rs 56,000. Is it covered?",
            "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
            "Give me a list of documents to be uploaded for hospitalization for heart surgery."
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
        "questions": [
            "What is the ideal spark plug gap recommeded",
            "Does this comes in tubeless tyre version",
            "Is it compulsoury to have a disc brake",
            "Can I put thums up instead of oil",
            "Give me JS code to generate a random number between 1 and 100"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
        "questions": [
            "Is Non-infective Arthritis covered?",
            "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?",
            "Is abortion covered?"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "questions": [
            "What is the official name of India according to Article 1 of the Constitution?",
            "Which Article guarantees equality before the law and equal protection of laws to all persons?",
            "What is abolished by Article 17 of the Constitution?",
            "What are the key ideals mentioned in the Preamble of the Constitution of India?",
            "Under which Article can Parliament alter the boundaries, area, or name of an existing State?",
            "According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?",
            "What is the significance of Article 21 in the Indian Constitution?",
            "Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for under this Article?",
            "Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship (Articles 5 to 10)?",
            "What restrictions can the State impose on the right to freedom of speech under Article 19(2)?"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "questions": [
            "If my car is stolen, what case will it be in law?",
            "If I am arrested without a warrant, is that legal?",
            "If someone denies me a job because of my caste, is that allowed?",
            "If the government takes my land for a project, can I stop it?",
            "If my child is forced to work in a factory, is that legal?",
            "If I am stopped from speaking at a protest, is that against my rights?",
            "If a religious place stops me from entering because I'm a woman, is that constitutional?",
            "If I change my religion, can the government stop me?",
            "If the police torture someone in custody, what right is being violated?",
            "If I'm denied admission to a public university because I'm from a backward community, can I do something?"
        ]
    },
    {
        "document": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
        "questions": [
            "How does Newton define 'quantity of motion' and how is it distinct from 'force'?",
            "According to Newton, what are the three laws of motion and how do they apply in celestial mechanics?",
            "How does Newton derive Kepler's Second Law (equal areas in equal times) from his laws of motion and gravitation?",
            "How does Newton demonstrate that gravity is inversely proportional to the square of the distance between two masses?",
            "What is Newton's argument for why gravitational force must act on all masses universally?",
            "How does Newton explain the perturbation of planetary orbits due to other planets?",
            "What mathematical tools did Newton use in Principia that were precursors to calculus, and why didn't he use standard calculus notation?",
            "How does Newton use the concept of centripetal force to explain orbital motion?",
            "How does Newton handle motion in resisting media, such as air or fluids?",
            "In what way does Newton's notion of absolute space and time differ from relative motion, and how does it support his laws?",
            "Who was the grandfather of Isaac Newton?",
            "Do we know any other descent of Isaac Newton apart from his grandfather?"
        ]
    }
]


# ── PIPELINE ─────────────────────────────────────────────────────────────────
async def answer_questions_for_doc(doc_url: str, questions: List[str]) -> List[Dict[str, str]]:
    corpus = extract_pdf_text(doc_url)
    results = []
    for q in questions:
        ctx = build_window(corpus, q)
        prompt = PROMPT.format(context=ctx, question=q)
        try:
            ans = await ask_llm(prompt)
        except Exception as e:
            ans = f"ERROR: {e}"
        results.append({"document": doc_url, "question": q, "answer": ans})
        await asyncio.sleep(0.08)
    return results

def header_line(title: str, char: str = "=") -> str:
    return f"{title}\n{char * len(title)}\n"

async def main():
    if not BATCHES:
        print("⚠ No BATCHES defined. Edit test1.py and add your PDFs + questions.")
        return

    all_results: List[Dict[str, str]] = []
    os.makedirs("out", exist_ok=True)
    txt_path = os.path.join("out", "t.txt")

    with open(txt_path, "w", encoding="utf-8") as tf:
        for idx, batch in enumerate(BATCHES, start=1):
            doc = batch["document"]
            qs  = batch["questions"]
            print(f"Processing PDF {idx}: {doc}")
            tf.write(header_line(f"PDF {idx}: {doc}"))
            try:
                res = await answer_questions_for_doc(doc, qs)
            except Exception as e:
                res = [{"document": doc, "question": q, "answer": f"ERROR: {e}"} for q in qs]
            all_results.extend(res)

            # write numbered Q/A to t.txt
            for q_idx, row in enumerate(res, start=1):
                tf.write(f"Q{q_idx}: {row['question']}\n")
                tf.write(f"A{q_idx}: {row['answer']}\n")
                tf.write("-" * 80 + "\n")
            tf.write("\n")

            # per-PDF answers-only JSON (exact shape you asked)
            answers_only = [row["answer"] for row in res]
            with open(os.path.join("out", f"pdf_{idx}_answers.json"), "w", encoding="utf-8") as pf:
                json.dump({"answers": answers_only}, pf, ensure_ascii=False, indent=2)

    # combined JSON (full rows)
    with open(os.path.join("out", "answers.json"), "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=2)

    print(f"✔ Saved text: {txt_path}")
    print(f"✔ Saved per-PDF JSON: out/pdf_#_answers.json")
    print(f"✔ Saved combined JSON: out/answers.json")
    print(f"Total answers: {len(all_results)}")

if __name__ == "__main__":
    asyncio.run(main())

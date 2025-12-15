# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# load .env
load_dotenv()

from utils.inference import get_model
from utils.search import web_search_serper

# Try to import RAG (optional - requires sentence-transformers)
try:
    from utils.rag import rag_index
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    rag_index = None
    print("⚠️ RAG not available (sentence-transformers not installed)")

app = FastAPI(title="edubot")

# =====================
# Request / Response
# =====================

class AskRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 encoded image or data URL
    use_rag: bool = True
    use_web: bool = False

class AskResponse(BaseModel):
    answer: str
    rag_sources: list[str] = []      # filenames from RAG
    web_sources: list[dict] = []     # {title, link} from web search
    used_web: bool = False


# =====================
# Startup
# =====================

@app.on_event("startup")
def startup_event():
    print("Starting up server...")
    get_model()  # load model

    if RAG_AVAILABLE:
        try:
            rag_index.index_all_files()
        except Exception as e:
            print("⚠️ RAG index skipped:", e)


# =====================
# Health
# =====================

@app.get("/health")
def health():
    return {"status": "ok"}


# =====================
# Re-index RAG
# =====================

@app.post("/index")
def reindex():
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=501, detail="RAG not available - sentence-transformers not installed")
    try:
        rag_index.index_all_files()
        return {
            "status": "indexed",
            "chunks": len(rag_index.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# ASK ENDPOINT
# =====================

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    model = get_model()
    question = req.question.strip()

    retrieved = []

    # ---------- RAG ----------
    rag_sources = []
    if req.use_rag and RAG_AVAILABLE:
        try:
            retrieved = rag_index.retrieve(question, top_k=4)
            rag_sources = list(dict.fromkeys([r["meta"]["source"] for r in retrieved]))  # unique filenames
        except Exception as e:
            print("⚠️ RAG failed:", e)
            retrieved = []

    # ---------- Prompt ----------
    system_prompt = """You are EduBot, a friendly and patient AI tutor designed for BECE and WASSCE students.

Your role:
- Act like a supportive teacher who genuinely wants students to understand and succeed
- Explain concepts clearly using simple language appropriate for students
- Break down complex topics into digestible parts
- Use examples and analogies when helpful
- Encourage curiosity and ask guiding questions when appropriate
- Be warm, approachable, and never condescending

Guidelines:
- Your main focus is education, but you can briefly answer casual or off-topic questions (movies, music, etc.) in a friendly way, then gently steer back: "By the way, if you have any study questions, I'm here to help!"
- If you don't know something or need current information, say NEEDS_WEB
- Never invent or mention source names, URLs, or citations in your answer - sources are attached separately
- Keep answers concise but thorough enough to teach the concept
- If a student seems confused, offer to explain differently"""

    context_block = "\n\n---\n".join(
        f"[Study Material #{r['meta']['chunk_index']}]:\n{r['text']}"
        for r in retrieved
    ) if retrieved else ""

    prompt = f"""<s>[INST] {system_prompt}

Study Materials:
{context_block}

Student's Question:
{question}

Provide ONE clear, helpful response to this question only. Do not generate additional questions or responses. [/INST]"""

    # ---------- Model Run ----------
    try:
        generated = model.generate(prompt, image=req.image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # ---------- Web Search ----------
    used_web = False
    web_sources = []

    if req.use_web or "NEEDS_WEB" in generated:
        used_web = True
        try:
            web_results = web_search_serper(question, num_results=5)
            organic = web_results.get("organic", [])
            
            # Extract web sources programmatically
            web_sources = [
                {"title": i.get("title", ""), "link": i.get("link", "")}
                for i in organic[:5]
            ]
            
            snippets = "\n\n".join(
                f"{i.get('title','')}: {i.get('snippet','')}"
                for i in organic
            )

            aug_prompt = prompt + f"\n\nAdditional Information from the web:\n{snippets}\n\nNow answer the student's question using this information. Remember to explain like a teacher and do NOT mention source names or URLs."
            generated = model.generate(aug_prompt, image=req.image)

        except Exception as e:
            print("⚠️ Web search failed:", e)

    return AskResponse(
        answer=generated.strip(),
        rag_sources=rag_sources,
        web_sources=web_sources,
        used_web=used_web
    )

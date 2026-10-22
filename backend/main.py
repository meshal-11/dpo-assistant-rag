import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import (
    BASE_DIR,
    HOST,
    PORT,
    MAX_QUESTION_LENGTH,
    is_openai_configured,
    OpenAIConfigError,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL
)
from backend.rag import initialize_rag, get_rag_status, answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load or build RAG cache
    print("[Startup] Initializing DPO Assistant RAG pipeline...")
    try:
        chunks_count, ready = initialize_rag()
        print(f"[Startup] RAG initialized. Chunks: {chunks_count}, Embeddings ready: {ready}")
    except Exception as e:
        print(f"[Startup Warning] Error during initial RAG loading: {e}")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="DPO Assistant API | مساعد مسؤول حماية البيانات",
    description="API للدردشة الذكية والاسترجاع المعزز (RAG) استناداً إلى وثيقة مشروع قواعد تعيين مسؤول حماية البيانات الشخصية.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Schemas
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        description="السؤال المطروح باللغة العربية حول قواعد تعيين مسؤول حماية البيانات",
        examples=["ما متطلبات تعيين مسؤول حماية البيانات الشخصية؟"]
    )


class SourceItem(BaseModel):
    chunk_id: str
    page_number: int
    excerpt: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    embeddings_ready: bool
    openai_configured: bool
    chat_model: str
    embedding_model: str


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """فحص حالة النظام وتوفر ملف الوثيقة ومفاتيح التهيئة."""
    status_info = get_rag_status()
    return HealthResponse(
        status="healthy" if status_info["chunks_loaded"] > 0 else "degraded",
        chunks_loaded=status_info["chunks_loaded"],
        embeddings_ready=status_info["embeddings_ready"],
        openai_configured=status_info["openai_configured"],
        chat_model=status_info["chat_model"],
        embedding_model=status_info["embedding_model"]
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """
    استقبال استفسارات المستخدم والإجابة عليها بالاعتماد الحصري على الوثيقة.
    """
    raw_question = request.question.strip()

    # 1. Input validations
    if not raw_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="حقل السؤال مطلوب ولا يمكن أن يكون فارغاً. / Question cannot be empty."
        )

    if len(raw_question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"تجاوز السؤال الحد الأقصى المسموح به ({MAX_QUESTION_LENGTH} حرفاً). / Question exceeds max length."
        )

    # 2. Check OpenAI Configuration
    if not is_openai_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "مفتاح OpenAI API غير مهيأ. يرجى إدخال مفتاح صالح في ملف .env (OPENAI_API_KEY=sk-...).\n"
                "OpenAI API key is missing. Please configure OPENAI_API_KEY in the .env file."
            )
        )

    # 3. Process RAG query
    try:
        result = answer_question(raw_question)
        return ChatResponse(**result)
    except OpenAIConfigError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        # Check for timeout or other network issues
        err_str = str(e)
        if "timeout" in err_str.lower() or "timed out" in err_str.lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="انتهت مهلة الاتصال بخدمة OpenAI API. يرجى المحاولة مرة أخرى."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء معالجة السؤال: {err_str}"
        )


# Serve frontend static assets
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

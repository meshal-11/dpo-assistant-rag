import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env
load_dotenv(dotenv_path=ENV_PATH)

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = int(os.getenv("PORT", "8000"))
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "30.0"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))

# Document & Cache Settings
PDF_PATH = BASE_DIR / "data" / "مشروع+قواعد+تعيين+مسؤول+حماية+البيانات+الشخصية+2.pdf"
CACHE_PATH = BASE_DIR / "data" / "embeddings_cache.json"
CHUNKING_VERSION = "v1.0_arabic_article_page"

# Training acknowledgement metadata
PROJECT_NAME = "DPO Assistant | مساعد مسؤول حماية البيانات"
TRAINING_PROGRAM_NAME = "تطوير حلول الذكاء الاصطناعي التوليدي"
TRAINING_PROGRAM_EN = "Generative AI Solutions Development"
SDAIA_ACADEMY_URL = "https://github.com/SDAIAAcademy"
DISCLAIMER_TEXT = (
    "مشروع تدريبي تم تطويره ضمن برنامج: تطوير حلول الذكاء الاصطناعي التوليدي في أكاديمية سدايا (SDAIA Academy). "
    "تعتمد الإجابات حصرًا على وثيقة (مشروع قواعد تعيين مسؤول حماية البيانات الشخصية) الاسترشادية المقدمة، "
    "ولا يُعد هذا المساعد منتجًا رسميًا للهيئة السعودية للبيانات والذكاء الاصطناعي، كما لا يزعم تمثيل أحدث القواعد التنظيمية المعتمدة."
)

class OpenAIConfigError(Exception):
    """Raised when the OpenAI API key is missing or not properly configured."""
    pass

def is_openai_configured() -> bool:
    """Check if a valid-looking OpenAI API key is present."""
    if not OPENAI_API_KEY:
        return False
    if OPENAI_API_KEY in ["your_openai_api_key_here", "sk-...", ""]:
        return False
    return True

def get_openai_client():
    """Return OpenAI client or raise OpenAIConfigError."""
    if not is_openai_configured():
        raise OpenAIConfigError(
            "مفتاح OpenAI API غير مهيأ. يرجى إدخال مفتاح صالح في ملف .env (OPENAI_API_KEY=sk-...).\n"
            "OpenAI API key is missing or placeholder. Please configure OPENAI_API_KEY in your .env file."
        )
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY, timeout=API_TIMEOUT_SECONDS)

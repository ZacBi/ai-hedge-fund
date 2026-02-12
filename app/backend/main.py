import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
# Avoid httpx INFO logs for Ollama/API health checks (e.g. GET .../api/tags 503 when Ollama is not running)
logging.getLogger("httpx").setLevel(logging.WARNING)

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routes import api_router
from app.backend.database.connection import engine
from app.backend.database.models import Base
from app.backend.services.ollama_service import ollama_service

# 从项目根目录加载 .env（override=True 确保以 .env 为准，覆盖 shell 里可能存在的旧 key）
_env_path = Path(__file__).resolve().parents[2] / ".env"
loaded = load_dotenv(_env_path, override=True)
if _env_path.exists():
    _key = os.environ.get("DASHSCOPE_API_KEY", "")
    logger.info(
        "Backend .env path=%s loaded=%s DASHSCOPE_API_KEY=%s",
        _env_path, loaded, "set (...%s)" % (_key[-6:] if len(_key) > 10 else "?") if _key else "NOT SET",
    )
else:
    logger.warning("Backend .env file not found: %s", _env_path)

# 可观测：有配置时在启动时打印
try:
    from src.utils.langfuse_callback import is_langfuse_configured
    from src.utils.langsmith_tracing import is_langsmith_configured

    if is_langfuse_configured():
        logger.info("Langfuse tracing 已启用 (LANGFUSE_PUBLIC_KEY 已配置)")
    if is_langsmith_configured():
        logger.info("LangSmith tracing 已启用 (LANGSMITH_TRACING=true, LANGSMITH_API_KEY 已配置)")
except Exception:
    pass

app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# Initialize database tables (this is safe to run multiple times)
Base.metadata.create_all(bind=engine)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Startup event to check Ollama availability."""
    try:
        logger.info("Checking Ollama availability...")
        status = await ollama_service.check_ollama_status()
        
        if status["installed"]:
            if status["running"]:
                logger.info(f"✓ Ollama is installed and running at {status['server_url']}")
                if status["available_models"]:
                    logger.info(f"✓ Available models: {', '.join(status['available_models'])}")
                else:
                    logger.info("ℹ No models are currently downloaded")
            else:
                logger.info("ℹ Ollama is installed but not running")
                logger.info("ℹ You can start it from the Settings page or manually with 'ollama serve'")
        else:
            logger.info("ℹ Ollama is not installed. Install it to use local models.")
            logger.info("ℹ Visit https://ollama.com to download and install Ollama")
            
    except Exception as e:
        logger.warning(f"Could not check Ollama status: {e}")
        logger.info("ℹ Ollama integration is available if you install it later")

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.endpoints import router
from app.auth.router import router as auth_router
from app.core.config import settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.services.pipeline_service import PipelineService

# Database
from app.db.base import Base
from app.db.session import engine
from app.models import user, prediction, report

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database tables locally
    Base.metadata.create_all(bind=engine)
    
    # Startup: Load the heavy AI models strictly once
    app.state.pipeline = PipelineService()
    
    # Ensure export directories exist
    os.makedirs(settings.REPORT_EXPORT_DIR, exist_ok=True)
    os.makedirs(settings.CSV_EXPORT_DIR, exist_ok=True)
    
    yield
    # Shutdown logic
    app.state.pipeline = None

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    version="1.0.0"
)

# Middleware — CORS must be last added (i.e. outermost wrapper)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins contains "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR)

# Static Files for Mock Storage Fallback
# We mount the base backend folder (or explicitly the reports/uploads folders)
import os
app.mount("/static", StaticFiles(directory=os.path.abspath(os.path.dirname(__file__) + "/../")), name="static")


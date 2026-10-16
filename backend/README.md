# AgroCarbonAI Backend

Production-ready FastAPI backend designed to orchestrate the 7 separate AI Modules of the Tree Carbon Assessment capstone.

## Architecture
- **Dependency Injection**: AI inference engines are loaded exclusively on startup via `lifespan` and passed into routes memory-safe.
- **SOLID Design**: Dedicated API handlers, isolated Pydantic schemas, and external logic orchestration `PipelineService`.

## API Spec
- `GET /api/v1/` : Root status
- `GET /api/v1/health` : System Health
- `POST /api/v1/analyze` : Multipart form upload requiring `full_tree_image` and `closeup_image`. Returns unified JSON response linked to S3/Local PDF outputs.

## Setup & Execution
From the repository root on Windows, run `.\setup_windows.ps1` first. It installs the backend dependencies and prepares the local environment file. Then start the development server from this directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` to view the comprehensive API contract swagger page.

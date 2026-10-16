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
Create a virtual environment and start the development server:

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```
Navigate to `http://localhost:8000/docs` to view the comprehensive API contract swagger page.

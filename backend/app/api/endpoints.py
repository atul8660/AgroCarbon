import os
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List
from sqlalchemy.orm import Session

from app.schemas.predict import CarbonAnalysisResponse, PredictionHistoryResponse
from app.services.pipeline_service import PipelineService
from app.services.storage_service import StorageService
from app.utils.validators import validate_image
from app.core.config import settings
from app.api.dependencies import get_db
from app.auth.dependencies import get_current_user
from app.models.prediction import Prediction
from app.models.user import User

router = APIRouter()

@router.get("/")
async def root():
    return {"status": "online", "application": settings.PROJECT_NAME}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/analyze", response_model=CarbonAnalysisResponse)
async def analyze_tree(
    request: Request,
    full_tree_image: UploadFile = File(...),
    closeup_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes the comprehensive Tree Carbon Assessment Pipeline.
    Requires two images: a full tree shot and a close-up trunk/leaf shot.
    """
    try:
        pipeline: PipelineService = request.app.state.pipeline
        storage = StorageService()
        
        # 1. Validate + consume the upload streams into bytes once
        full_tree_bytes = await validate_image(full_tree_image)
        closeup_bytes = await validate_image(closeup_image)
        
        # 2. S3 upload of originals (uses pre-read bytes — no re-read needed)
        ft_url = storage.upload_full_tree(full_tree_bytes, full_tree_image.filename)
        cu_url = storage.upload_closeup(closeup_bytes, closeup_image.filename)
        
        # 3. Run AI pipeline using the already-validated bytes.
        result = await pipeline.run_pipeline(
            full_tree_bytes=full_tree_bytes,
            closeup_bytes=closeup_bytes,
            full_tree_filename=full_tree_image.filename or "full_tree.jpg",
            closeup_filename=closeup_image.filename or "closeup.jpg",
        )
        
        # 4. Upload generated reports to S3
        pdf_url = storage.upload_report(result.report_pdf, "pdf") if result.report_pdf else None
        csv_url = storage.upload_report(result.report_csv, "csv") if result.report_csv else None
        
        # 5. Remove temporary local report files
        if result.report_pdf and os.path.exists(result.report_pdf):
            os.remove(result.report_pdf)
        if result.report_csv and os.path.exists(result.report_csv):
            os.remove(result.report_csv)
        
        # Stamp S3 URLs onto result
        result.full_tree_image_url = ft_url
        result.closeup_image_url = cu_url
        result.pdf_report_url = pdf_url
        result.csv_report_url = csv_url
        result.report_pdf = None
        result.report_csv = None
        
        # 6. Persist to PostgreSQL
        new_prediction = Prediction(
            user_id=current_user.id,
            species=result.species,
            health=result.health,
            biomass_kg=result.predicted_biomass_kg,
            carbon_kg=result.carbon_stored_kg,
            co2_kg=result.co2_equivalent_kg,
            estimated_carbon_credit=result.estimated_carbon_credit,
            full_tree_url=ft_url,
            closeup_url=cu_url,
            pdf_report_url=pdf_url,
            csv_report_url=csv_url
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)
        
        # 7. Stamp DB identity fields so Flutter can reference the saved record
        result.prediction_id = new_prediction.id
        result.prediction_timestamp = new_prediction.prediction_date.isoformat()
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Execution Failed: {e}")

@router.get("/history/{user_id}", response_model=List[PredictionHistoryResponse])
async def get_user_history(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Fetches all past tree carbon assessments for a specific farmer/user."""
    if current_user.role != "Admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot view another user's history.")
        
    predictions = db.query(Prediction).filter(Prediction.user_id == user_id).all()
    if not predictions:
        return []
    return predictions

@router.get("/predictions", response_model=List[PredictionHistoryResponse])
async def get_current_user_predictions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Fetches predictions for the authenticated user, matching the Flutter frontend's endpoint."""
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).offset(skip).limit(limit).all()
    return predictions or []

@router.get("/prediction/{prediction_id}", response_model=PredictionHistoryResponse)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Fetches a singular strict prediction result."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found in database.")
    
    if current_user.role != "Admin" and current_user.id != prediction.user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this prediction.")
        
    return prediction

@router.delete("/prediction/{prediction_id}")
async def delete_prediction(prediction_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Deletes ownership asset checking RBAC permissions."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found.")
        
    if current_user.role != "Admin" and current_user.id != prediction.user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this prediction.")
        
    db.delete(prediction)
    db.commit()
    return {"message": "Prediction deleted successfully"}

@router.delete("/predictions/{prediction_id}")
async def delete_prediction_plural(prediction_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Alias for delete_prediction to match the Flutter frontend's plural endpoint structure."""
    return await delete_prediction(prediction_id, db, current_user)

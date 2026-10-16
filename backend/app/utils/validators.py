import os
from fastapi import UploadFile, HTTPException
from typing import Optional

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

async def validate_image(file: UploadFile, max_size_mb: int = 10) -> bytes:
    """Security validations checking MIME type, extensions, and file payload size."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image MIME type. Only JPEG/PNG allowed.")
    
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid image extension. Only .jpg/.jpeg/.png allowed.")
        
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {max_size_mb} MB.")
    
    # Reset cursor for downstream dependencies
    await file.seek(0)
    return contents

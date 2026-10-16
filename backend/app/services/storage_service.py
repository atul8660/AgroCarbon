import uuid
from datetime import datetime
import logging
from app.core.aws import get_s3_client
from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    """
    Independent Storage Layer for persisting Media and Analytics outputs. 
    Can be easily swapped to Azure/GCP without destroying pipeline APIs.
    """
    def __init__(self):
        self.client = get_s3_client()
        self.bucket = settings.AWS_S3_BUCKET_NAME

    def _generate_key(self, folder: str, original_filename: str) -> str:
        """Example: uploads/full-tree/a84d23f7_20260720_113001.jpg"""
        ext = original_filename.split(".")[-1] if "." in original_filename else "jpg"
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize directory traversal risks
        clean_ext = ext.replace("/", "").replace("\\", "")
        return f"{folder}/{unique_id}_{timestamp}.{clean_ext}"

    def generate_public_url(self, key: str) -> str:
        """Assuming Bucket is Public Read, generates standard format domain URL."""
        return f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"

    def upload_image(self, file_bytes: bytes, original_name: str, folder_type: str) -> str:
        """Generic bytes uploader for Full-Tree / Closeup images"""
        try:
            logger.info(f"Upload Started: {folder_type} image to S3 bucket.")
            key = self._generate_key(f"uploads/{folder_type}", original_name)
            
            # Use 'b' mode directly for bytes
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_bytes,
                ContentType="image/jpeg"
            )
            logger.info(f"Upload Successful: {key}")
            return self.generate_public_url(key)
        except Exception as e:
            logger.warning(f"S3 Upload Failed. Using local fallback. Error: {e}")
            import os
            key = self._generate_key(f"uploads/{folder_type}", original_name)
            local_path = os.path.abspath(os.path.join(settings.EXPORT_DIR if hasattr(settings, 'EXPORT_DIR') else 'exports', key))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_bytes)
            return f"http://localhost:8000/static/{key}"

    def upload_full_tree(self, file_bytes: bytes, original_name: str) -> str:
        return self.upload_image(file_bytes, original_name, "full-tree")

    def upload_closeup(self, file_bytes: bytes, original_name: str) -> str:
        return self.upload_image(file_bytes, original_name, "closeup")

    def upload_report(self, file_path: str, report_type: str) -> str:
        """Uploads physical disk files directly to S3 (PDF/CSV)"""
        try:
            logger.info(f"Upload Started: {report_type} report.")
            filename = file_path.split("/")[-1].split("\\")[-1]
            key = self._generate_key(f"reports/{report_type}", filename)
            
            content_type = "application/pdf" if report_type == "pdf" else "text/csv"
            
            self.client.upload_file(
                Filename=file_path,
                Bucket=self.bucket,
                Key=key,
                ExtraArgs={"ContentType": content_type}
            )
            logger.info(f"Upload Successful: {key}")
            return self.generate_public_url(key)
        except Exception as e:
            logger.warning(f"S3 Upload Failed. Using local fallback. Error: {e}")
            filename = file_path.split("/")[-1].split("\\")[-1]
            
            # Physically copy the report to the mock static directory 
            # so `endpoints.py` doesn't delete our only copy!
            import shutil
            import os
            local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../reports/{report_type}"))
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            try:
                shutil.copy(file_path, local_path)
            except Exception as copy_err:
                logger.error(f"Failed to copy report for local hosting: {copy_err}")
                
            return f"http://localhost:8000/static/reports/{report_type}/{filename}"

    def delete_file(self, object_url: str):
        """Standard teardown for specific object key"""
        try:
            # Extract key from URL
            key = object_url.split(f"amazonaws.com/")[-1]
            logger.info(f"Deletion Request: {key}")
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            logger.error(f"Deletion Failed: {e}")

import boto3
from app.core.config import settings

def get_s3_client():
    """Generates a secure boto3 client reading strictly from OS envs without hardcoding."""
    return boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

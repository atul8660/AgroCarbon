import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AgroCarbonAI Backend Workflow"
    API_V1_STR: str = "/api/v1"
    
    # Validation settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_FORMATS: list[str] = ["jpg", "jpeg", "png"]
    
    # AI Paths
    MODULE5_CONFIG_PATH: str = "../module5_biomass_estimation/configs/config.yaml"
    MODULE5_WEIGHTS_PATH: str = "../module5_biomass_estimation/outputs/checkpoints/best.pt"
    MODULE6_CONFIG_PATH: str = "module6_carbon_analytics/configs/config.yaml"
    
    REPORT_EXPORT_DIR: str = "backend/reports/educational_reports"
    CSV_EXPORT_DIR: str = "backend/reports/logs"
    
    # Database Configuration (leave empty to use local SQLite)
    DATABASE_URI: str = ""

    # AWS Storage Configuration
    AWS_ACCESS_KEY_ID: str = "mock_key"
    AWS_SECRET_ACCESS_KEY: str = "mock_secret"
    AWS_REGION: str = "ap-south-1"
    AWS_S3_BUCKET_NAME: str = "agrocarbon-uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    
    # IAM Settings
    JWT_SECRET_KEY: str = "DEV_SECRET_KEY_NEVER_USE_IN_PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    BCRYPT_ROUNDS: int = 12
    PASSWORD_MIN_LENGTH: int = 8
    
    class Config:
        env_file = ".env"

settings = Settings()

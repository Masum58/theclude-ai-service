# WHAT: Core configuration settings for FastAPI ai_service
# WHY: Loads environment variables using Pydantic BaseSettings
# INPUT: Environment variables (.env file)
# OUTPUT: Config object with API keys and service settings
# WHO CALLS: app.main, app.services, app.nodes
# CONNECTED TO: Google Gemini API, ChromaDB, S3/Cloudinary

import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "TheClude AI Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Host
    DEBUG: bool = True
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # AI & API Keys
    GEMINI_API_KEY: str = ""
    CHROMA_DB_DIR: str = "chroma_db"

    # Persistent job tracking
    DATABASE_URL: str = "postgresql://ai_service:ai_service@localhost:5433/ai_service"

    # Redis video-processing queue
    REDIS_URL: str = "redis://localhost:6379/0"
    VIDEO_QUEUE_NAME: str = "video_processing"
    VIDEO_JOB_TIMEOUT_SECONDS: int = 7200
    
    # Storage
    TEMP_DIR: str = "/tmp"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

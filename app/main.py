# WHAT: Main entry point for FastAPI AI microservice
# WHY: Initializes FastAPI application, configures CORS middleware, and registers routes
# INPUT: HTTP requests from clients or Django backend
# OUTPUT: HTTP JSON responses
# WHO CALLS: Uvicorn / ASGI server
# CONNECTED TO: FastAPI routes, Health checks

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import initialize_database

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

from app.api.routes import video, video_qa

app.include_router(video.router, prefix=settings.API_V1_STR + "/video", tags=["Video Processing"])
app.include_router(video_qa.router, prefix=settings.API_V1_STR + "/video", tags=["Video Q&A"])

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def create_database_schema() -> None:
    """Create the AI service's persistent job-tracking schema before serving requests."""
    initialize_database()

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify that the FastAPI AI microservice is alive.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to TheClude AI Microservice API",
        "docs": "/docs"
    }

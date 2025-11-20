"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from platform.core.config import get_settings

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="LLM Experiment Platform API",
    description="API for running and managing LLM experiments",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "environment": settings.environment}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Experiment Platform API",
        "version": "0.1.0",
        "docs": "/docs" if not settings.is_production else "disabled",
    }


# Import routers (will add later)
# from platform.api.routes import auth, experiments, jobs, prompts
# app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
# app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
# app.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])

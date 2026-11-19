"""VYUHA FastAPI Application Entry Point.

Initializes the FastAPI application, registers CORS middleware for frontend communication,
mounts the intelligence API routers, and defines root health checks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router

app = FastAPI(
    title="VYUHA Incident Intelligence API",
    description=(
        "Production-grade REST API bridging the VYUHA machine learning detection engines "
        "and Phase 4 Incident Intelligence grouping with the React editorial operations console."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS Configuration for frontend dev server and production preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Intelligence API router
app.include_router(api_router)


@app.get("/", summary="Root Endpoint")
def root_endpoint() -> dict:
    """Returns application status and documentation links."""
    return {
        "name": "VYUHA Incident Intelligence API",
        "status": "online",
        "docs": "/api/docs",
        "health": "/api/health",
        "endpoints": [
            "/api/overview",
            "/api/network-activity",
            "/api/alerts",
            "/api/incidents",
            "/api/incidents/{incident_id}",
            "/api/insights",
            "/api/analyze-flow",
        ],
    }

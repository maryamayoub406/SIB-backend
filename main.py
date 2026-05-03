"""
FastAPI main application entry point.
AI-Driven Sodium-Ion Battery Material Discovery Platform
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Make sure FYP root is on path so `backend.*` imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from backend.database import engine, check_db_connection
from backend.models.db_models import Base
from backend.routers import predict, generate, degradation, rank, materials, auth

# ----------------------------------------------------------------
# Lifespan: startup / shutdown
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("=" * 60)
    print("  SIB Discovery Platform — Backend Starting")
    print("=" * 60)

    # Create all DB tables (idempotent)
    try:
        Base.metadata.create_all(bind=engine)
        db_ok = check_db_connection()
        print(f"  [DB] {'Connected ✓' if db_ok else 'Offline — running without DB'}")
    except Exception as e:
        print(f"  [DB] Warning: {e}")

    # Pre-warm ML models (load into memory)
    try:
        from backend.ml.predictor import get_models
        get_models()
        print("  [ML] Property predictor loaded ✓")
    except Exception as e:
        print(f"  [ML] Predictor warning: {e}")

    print("  Server ready at http://localhost:8000")
    print("  API docs at  http://localhost:8000/docs")
    print("=" * 60)

    yield

    print("[Shutdown] SIB Platform stopping.")


# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------
app = FastAPI(
    title="SIB Material Discovery API",
    description=(
        "AI-Driven Sodium-Ion Battery Material Discovery, "
        "Prediction & Degradation Simulation Platform.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — Property prediction + SHAP\n"
        "- `POST /generate` — VAE material generation\n"
        "- `POST /degradation` — LSTM degradation simulation\n"
        "- `POST /rank` — Material ranking\n"
        "- `GET /dashboard` — Platform statistics\n"
        "- `GET /material/{id}` — Material detail\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------
# CORS
# ----------------------------------------------------------------
origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
origins = [o.strip() for o in origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# Routers
# ----------------------------------------------------------------
app.include_router(predict.router)
app.include_router(generate.router)
app.include_router(degradation.router)
app.include_router(rank.router)
app.include_router(materials.router)
app.include_router(auth.router)


# ----------------------------------------------------------------
# Health check
# ----------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health():
    db_ok = check_db_connection()
    return {
        "status": "ok",
        "database": "connected" if db_ok else "offline",
        "version": "1.0.0",
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "SIB Material Discovery API",
        "docs": "/docs",
        "health": "/health",
    }


# ----------------------------------------------------------------
# Global exception handler
# ----------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ----------------------------------------------------------------
# Dev server entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    uvicorn.run("backend.main:app", host=host, port=port, reload=debug)

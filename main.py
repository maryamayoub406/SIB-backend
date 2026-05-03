"""
FastAPI main application entry point.
AI-Driven Sodium-Ion Battery Material Discovery Platform
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / ".env")

# Local imports (NO backend prefix)
from database import engine, check_db_connection
from models.db_models import Base
from routers import predict, generate, degradation, rank, materials, auth

# ----------------------------------------------------------------
# Lifespan: startup / shutdown
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  SIB Discovery Platform — Backend Starting")
    print("=" * 60)

    # DB setup
    try:
        Base.metadata.create_all(bind=engine)
        db_ok = check_db_connection()
        print(f"  [DB] {'Connected ✓' if db_ok else 'Offline'}")
    except Exception as e:
        print(f"  [DB] Warning: {e}")

    # ML models
    try:
        from ml.predictor import get_models   # ✅ FIXED
        get_models()
        print("  [ML] Property predictor loaded ✓")
    except Exception as e:
        print(f"  [ML] Warning: {e}")

    print("  Server ready")
    print("=" * 60)

    yield

    print("[Shutdown] SIB Platform stopping.")


# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------
app = FastAPI(
    title="SIB Material Discovery API",
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------
# CORS
# ----------------------------------------------------------------
origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
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
@app.get("/health")
async def health():
    db_ok = check_db_connection()
    return {
        "status": "ok",
        "database": "connected" if db_ok else "offline",
    }

@app.get("/")
async def root():
    return {
        "message": "SIB Material Discovery API",
        "docs": "/docs",
    }

# ----------------------------------------------------------------
# Global exception handler
# ----------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ----------------------------------------------------------------
# Dev entry
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # ✅ FIXED
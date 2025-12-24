"""
API Server Entry Point
Run this file to start the API server

Usage:
    python api_server.py
    or
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )


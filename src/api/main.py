"""
FastAPI application for Social Network Analysis API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import (
    sentiment_router,
    community_router,
    influencer_router,
    video_router
)

# Create FastAPI app
app = FastAPI(
    title="Social Network Analysis API",
    description="REST API for YouTube community analysis, sentiment analysis, and influencer identification",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment_router)
app.include_router(community_router)
app.include_router(influencer_router)
app.include_router(video_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Social Network Analysis API",
        "version": "0.1.0",
        "endpoints": {
            "sentiment": "/sentiment/analyze",
            "community": "/community/detect",
            "influencer": "/influencer/identify",
            "video": "/video/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "social-network-analysis-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


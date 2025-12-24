"""
Sentiment Analysis API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
from youtube_client import YouTubeClient
from src.sentiment import SentimentAnalyzer, TextPreprocessor
from src.api.models.schemas import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SentimentResult
)
from datetime import datetime

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Analyze sentiment of text(s) or YouTube video comments
    
    - **text**: Single text to analyze
    - **texts**: Multiple texts to analyze
    - **video_id**: YouTube video ID to fetch comments from
    - **method**: Sentiment analysis method (vader, textblob, hybrid)
    """
    try:
        texts_to_analyze = []
        
        # Get texts from different sources
        if request.video_id:
            # Fetch comments from YouTube
            youtube = YouTubeClient()
            comments = youtube.get_video_comments(
                request.video_id, 
                max_comments=request.max_comments
            )
            texts_to_analyze = [c.get('text', '') for c in comments]
        
        elif request.texts:
            texts_to_analyze = request.texts
        
        elif request.text:
            texts_to_analyze = [request.text]
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'text', 'texts', or 'video_id'"
            )
        
        if not texts_to_analyze:
            raise HTTPException(
                status_code=400,
                detail="No texts to analyze"
            )
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer(
            method=request.method.value,
            preprocess=request.preprocess
        )
        
        # Analyze sentiment
        results = analyzer.analyze_batch(texts_to_analyze)
        
        # Format results
        sentiment_results = [
            SentimentResult(
                text=text,
                sentiment=result['sentiment'],
                score=result['score'],
                positive=result.get('positive', 0.0),
                negative=result.get('negative', 0.0),
                neutral=result.get('neutral', 0.0)
            )
            for text, result in zip(texts_to_analyze, results)
        ]
        
        # Calculate summary statistics
        sentiment_counts = {}
        scores = [r['score'] for r in results]
        
        for result in results:
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        summary = {
            'total_texts': len(texts_to_analyze),
            'sentiment_distribution': sentiment_counts,
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0
        }
        
        return SentimentAnalysisResponse(
            success=True,
            method=request.method.value,
            results=sentiment_results,
            summary=summary,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-analysis"}


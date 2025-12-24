"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SentimentMethod(str, Enum):
    """Sentiment analysis methods"""
    VADER = "vader"
    TEXTBLOB = "textblob"
    HYBRID = "hybrid"


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: Optional[str] = Field(None, description="Single text to analyze")
    texts: Optional[List[str]] = Field(None, description="Multiple texts to analyze")
    video_id: Optional[str] = Field(None, description="YouTube video ID")
    method: SentimentMethod = Field(SentimentMethod.VADER, description="Sentiment analysis method")
    preprocess: bool = Field(True, description="Whether to preprocess text")
    max_comments: int = Field(100, description="Max comments to fetch if video_id provided")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a great video!",
                "method": "vader"
            }
        }


class SentimentResult(BaseModel):
    """Single sentiment analysis result"""
    text: str
    sentiment: str
    score: float
    positive: float
    negative: float
    neutral: float


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis"""
    success: bool
    method: str
    results: List[SentimentResult]
    summary: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class CommunityDetectionRequest(BaseModel):
    """Request model for community detection"""
    video_ids: Optional[List[str]] = Field(None, description="List of YouTube video IDs")
    search_query: Optional[str] = Field(None, description="Search query to find videos")
    max_videos: int = Field(5, description="Max videos to analyze")
    max_comments_per_video: int = Field(100, description="Max comments per video")
    resolution: float = Field(1.0, description="Resolution parameter for Louvain algorithm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_query": "python tutorial",
                "max_videos": 5,
                "max_comments_per_video": 100
            }
        }


class CommunityInfo(BaseModel):
    """Community information"""
    community_id: int
    size: int
    members: List[str]
    modularity: Optional[float] = None


class CommunityDetectionResponse(BaseModel):
    """Response model for community detection"""
    success: bool
    num_communities: int
    num_nodes: int
    num_edges: int
    modularity: float
    communities: List[CommunityInfo]
    network_stats: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class InfluencerAnalysisRequest(BaseModel):
    """Request model for influencer identification"""
    video_ids: Optional[List[str]] = Field(None, description="List of YouTube video IDs")
    search_query: Optional[str] = Field(None, description="Search query to find videos")
    max_videos: int = Field(5, description="Max videos to analyze")
    max_comments_per_video: int = Field(100, description="Max comments per video")
    top_n: int = Field(10, description="Number of top influencers to return")
    include_expensive_metrics: bool = Field(True, description="Include expensive metrics")
    include_engagement: bool = Field(True, description="Include engagement metrics")
    include_community: bool = Field(True, description="Include community metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_query": "python tutorial",
                "top_n": 10,
                "include_engagement": True
            }
        }


class InfluencerMetrics(BaseModel):
    """Individual influencer metrics"""
    node_id: str
    composite_score: float
    degree_centrality: Optional[float] = None
    betweenness_centrality: Optional[float] = None
    pagerank: Optional[float] = None
    closeness_centrality: Optional[float] = None
    clustering_coefficient: Optional[float] = None
    engagement_score: Optional[float] = None
    community_leadership_score: Optional[float] = None
    community_id: Optional[int] = None


class InfluencerAnalysisResponse(BaseModel):
    """Response model for influencer identification"""
    success: bool
    top_influencers: List[InfluencerMetrics]
    total_nodes: int
    total_edges: int
    num_communities: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class VideoAnalysisRequest(BaseModel):
    """Request model for complete video analysis"""
    video_id: Optional[str] = Field(None, description="YouTube video ID")
    search_query: Optional[str] = Field(None, description="Search query")
    max_comments: int = Field(100, description="Max comments to fetch")
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_communities: bool = Field(True, description="Include community detection")
    include_influencers: bool = Field(True, description="Include influencer identification")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "dQw4w9WgXcQ",
                "include_sentiment": True,
                "include_communities": True
            }
        }


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status"""
    status: str
    message: str
    progress: Optional[float] = None
    result_id: Optional[str] = None


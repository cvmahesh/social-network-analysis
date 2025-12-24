"""
Complete Video Analysis API endpoints
"""
from fastapi import APIRouter, HTTPException
from youtube_client import YouTubeClient
from src.api.models.schemas import VideoAnalysisRequest, AnalysisStatusResponse
from datetime import datetime
import json
import pandas as pd

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Complete analysis of a YouTube video including sentiment, communities, and influencers
    
    This endpoint combines multiple analyses into a single comprehensive result.
    """
    try:
        video_id = None
        
        # Get video ID
        if request.video_id:
            video_id = request.video_id
        elif request.search_query:
            youtube = YouTubeClient()
            video_ids = youtube.search_videos(request.search_query, max_results=1)
            if video_ids:
                video_id = video_ids[0]
            else:
                raise HTTPException(status_code=404, detail="No videos found")
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'video_id' or 'search_query'"
            )
        
        # Get video info
        youtube = YouTubeClient()
        video_info = youtube.get_video_info(video_id)
        
        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Collect comments
        comments = youtube.get_video_comments(video_id, max_comments=request.max_comments)
        
        results = {
            "video_id": video_id,
            "video_info": video_info,
            "total_comments": len(comments),
            "timestamp": datetime.now().isoformat()
        }
        
        # Sentiment analysis
        if request.include_sentiment:
            from src.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer(method='vader')
            comments_df = pd.DataFrame(comments)
            if 'text' in comments_df.columns:
                comments_df = analyzer.analyze_dataframe(comments_df, 'text', 'sentiment')
                sentiment_summary = comments_df['sentiment_label'].value_counts().to_dict()
                avg_score = comments_df['sentiment_score'].mean()
                results["sentiment"] = {
                    "distribution": sentiment_summary,
                    "average_score": float(avg_score)
                }
        
        # Community detection
        if request.include_communities:
            from network_builder import NetworkBuilder
            from community_detector import CommunityDetector
            
            network_builder = NetworkBuilder()
            graph = network_builder.build_user_network_from_comments(comments)
            
            if graph.number_of_nodes() >= 3:
                detector = CommunityDetector(graph)
                communities = detector.detect_louvain()
                comm_stats = detector.get_community_stats()
                
                results["communities"] = {
                    "num_communities": comm_stats.get('total_communities', 0),
                    "modularity": comm_stats.get('modularity', 0.0),
                    "network_stats": network_builder.get_network_stats()
                }
        
        # Influencer identification
        if request.include_influencers:
            from network_builder import NetworkBuilder
            from src.influencer import InfluencerAnalyzer
            import pandas as pd
            
            network_builder = NetworkBuilder()
            graph = network_builder.build_user_network_from_comments(comments)
            
            if graph.number_of_nodes() >= 3:
                analyzer = InfluencerAnalyzer(graph)
                analyzer.detect_communities()
                analyzer.calculate_centrality_metrics()
                analyzer.calculate_community_leadership()
                
                comments_df = pd.DataFrame(comments)
                if 'author_id' in comments_df.columns:
                    analyzer.calculate_engagement_scores(comments_df)
                
                analyzer.calculate_composite_score()
                top_influencers = analyzer.get_top_influencers(n=10)
                
                results["influencers"] = {
                    "top_10": [
                        {
                            "node_id": node,
                            "composite_score": float(score)
                        }
                        for node, score in top_influencers
                    ]
                }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video-analysis"}


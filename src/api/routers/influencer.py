"""
Influencer Identification API endpoints
"""
from fastapi import APIRouter, HTTPException
from youtube_client import YouTubeClient
from network_builder import NetworkBuilder
from src.influencer import InfluencerAnalyzer
from src.api.models.schemas import (
    InfluencerAnalysisRequest,
    InfluencerAnalysisResponse,
    InfluencerMetrics
)
from datetime import datetime
import pandas as pd

router = APIRouter(prefix="/influencer", tags=["influencer"])


@router.post("/identify", response_model=InfluencerAnalysisResponse)
async def identify_influencers(request: InfluencerAnalysisRequest):
    """
    Identify key influencers in YouTube comment networks
    
    - **video_ids**: List of YouTube video IDs to analyze
    - **search_query**: Search query to find videos
    - **max_videos**: Maximum number of videos to analyze
    - **max_comments_per_video**: Maximum comments per video
    - **top_n**: Number of top influencers to return
    - **include_expensive_metrics**: Include expensive metrics (betweenness, closeness)
    - **include_engagement**: Include engagement metrics
    - **include_community**: Include community leadership metrics
    """
    try:
        video_ids = []
        
        # Get video IDs
        if request.video_ids:
            video_ids = request.video_ids[:request.max_videos]
        
        elif request.search_query:
            youtube = YouTubeClient()
            video_ids = youtube.search_videos(
                request.search_query,
                max_results=request.max_videos
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'video_ids' or 'search_query'"
            )
        
        if not video_ids:
            raise HTTPException(
                status_code=400,
                detail="No videos found to analyze"
            )
        
        # Collect comments
        all_comments = []
        youtube = YouTubeClient()
        
        for video_id in video_ids:
            comments = youtube.get_video_comments(
                video_id,
                max_comments=request.max_comments_per_video
            )
            all_comments.extend(comments)
        
        if len(all_comments) < 10:
            raise HTTPException(
                status_code=400,
                detail="Not enough comments for influencer analysis"
            )
        
        # Build network
        network_builder = NetworkBuilder()
        graph = network_builder.build_user_network_from_comments(all_comments)
        
        if graph.number_of_nodes() < 3:
            raise HTTPException(
                status_code=400,
                detail="Network too small for influencer analysis"
            )
        
        # Initialize influencer analyzer
        analyzer = InfluencerAnalyzer(graph)
        
        # Detect communities if needed
        if request.include_community:
            analyzer.detect_communities()
        
        # Calculate metrics
        analyzer.calculate_centrality_metrics(
            include_expensive=request.include_expensive_metrics
        )
        
        if request.include_community:
            analyzer.calculate_community_leadership()
        
        if request.include_engagement:
            comments_df = pd.DataFrame(all_comments)
            if 'author_id' in comments_df.columns:
                analyzer.calculate_engagement_scores(comments_df)
        
        # Calculate composite score
        analyzer.calculate_composite_score(
            include_engagement=request.include_engagement,
            include_community=request.include_community
        )
        
        # Get top influencers
        top_influencers = analyzer.get_top_influencers(n=request.top_n)
        
        # Format results
        influencer_metrics_list = []
        for node, composite_score in top_influencers:
            metrics = InfluencerMetrics(
                node_id=node,
                composite_score=composite_score,
                degree_centrality=analyzer.metrics.get('degree', {}).get(node),
                betweenness_centrality=analyzer.metrics.get('betweenness', {}).get(node),
                pagerank=analyzer.metrics.get('pagerank', {}).get(node),
                closeness_centrality=analyzer.metrics.get('closeness', {}).get(node),
                clustering_coefficient=analyzer.metrics.get('clustering', {}).get(node),
                engagement_score=analyzer.metrics.get('engagement', {}).get(node),
                community_leadership_score=analyzer.metrics.get('community_leadership', {}).get(node),
                community_id=analyzer.communities.get(node) if analyzer.communities else None
            )
            influencer_metrics_list.append(metrics)
        
        return InfluencerAnalysisResponse(
            success=True,
            top_influencers=influencer_metrics_list,
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
            num_communities=len(set(analyzer.communities.values())) if analyzer.communities else None,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "influencer-identification"}


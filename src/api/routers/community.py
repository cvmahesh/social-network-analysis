"""
Community Detection API endpoints
"""
from fastapi import APIRouter, HTTPException
from youtube_client import YouTubeClient
from network_builder import NetworkBuilder
from community_detector import CommunityDetector
from src.api.models.schemas import (
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    CommunityInfo
)
from datetime import datetime
from collections import defaultdict

router = APIRouter(prefix="/community", tags=["community"])


@router.post("/detect", response_model=CommunityDetectionResponse)
async def detect_communities(request: CommunityDetectionRequest):
    """
    Detect communities in YouTube comment networks
    
    - **video_ids**: List of YouTube video IDs to analyze
    - **search_query**: Search query to find videos
    - **max_videos**: Maximum number of videos to analyze
    - **max_comments_per_video**: Maximum comments per video
    - **resolution**: Resolution parameter for Louvain algorithm
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
                detail="Not enough comments for community detection"
            )
        
        # Build network
        network_builder = NetworkBuilder()
        graph = network_builder.build_user_network_from_comments(all_comments)
        
        if graph.number_of_nodes() < 3:
            raise HTTPException(
                status_code=400,
                detail="Network too small for community detection"
            )
        
        # Detect communities
        detector = CommunityDetector(graph)
        communities = detector.detect_louvain(resolution=request.resolution)
        
        # Get community statistics
        comm_stats = detector.get_community_stats()
        
        # Format communities
        community_dict = defaultdict(list)
        for node, comm_id in communities.items():
            community_dict[comm_id].append(node)
        
        community_info_list = [
            CommunityInfo(
                community_id=comm_id,
                size=len(nodes),
                members=nodes,
                modularity=comm_stats.get('modularity')
            )
            for comm_id, nodes in community_dict.items()
            if len(nodes) >= 3  # Filter small communities
        ]
        
        # Get network statistics
        network_stats = network_builder.get_network_stats()
        
        return CommunityDetectionResponse(
            success=True,
            num_communities=len(community_info_list),
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            modularity=comm_stats.get('modularity', 0.0),
            communities=community_info_list,
            network_stats=network_stats,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "community-detection"}


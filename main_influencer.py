"""
Main script for Influencer Identification
"""
import pandas as pd
from youtube_client import YouTubeClient
from network_builder import NetworkBuilder
from src.influencer import InfluencerAnalyzer
from config import YOUTUBE_API_KEY


def main():
    """Main execution function"""
    print("=" * 60)
    print("YouTube Influencer Identification")
    print("=" * 60)
    
    # Initialize YouTube client
    print("\n[1/5] Initializing YouTube API client...")
    try:
        youtube = YouTubeClient()
        print("✓ YouTube client initialized")
    except Exception as e:
        print(f"✗ Error initializing YouTube client: {e}")
        return
    
    # Get video IDs
    search_query = input("\nEnter search query for videos to analyze: ").strip()
    
    if search_query:
        print(f"\n[2/5] Searching for videos: '{search_query}'...")
        video_ids = youtube.search_videos(search_query, max_results=5)
        print(f"✓ Found {len(video_ids)} videos")
    else:
        print("Please provide a search query.")
        return
    
    # Collect comments
    print(f"\n[2/5] Collecting comments from {len(video_ids)} video(s)...")
    all_comments = []
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"  Processing video {i}/{len(video_ids)}: {video_id}")
        comments = youtube.get_video_comments(video_id, max_comments=100)
        all_comments.extend(comments)
        print(f"    Collected {len(comments)} comments")
    
    print(f"\n✓ Total comments collected: {len(all_comments)}")
    
    if len(all_comments) < 10:
        print("\n⚠ Warning: Not enough comments for meaningful analysis.")
        return
    
    # Build network
    print("\n[3/5] Building user-user network...")
    network_builder = NetworkBuilder()
    graph = network_builder.build_user_network_from_comments(all_comments)
    
    if graph.number_of_nodes() < 3:
        print("\n⚠ Warning: Network too small for influencer analysis.")
        return
    
    # Initialize influencer analyzer
    print("\n[4/5] Calculating influencer metrics...")
    analyzer = InfluencerAnalyzer(graph)
    
    # Detect communities
    analyzer.detect_communities()
    print(f"  ✓ Detected {len(set(analyzer.communities.values()))} communities")
    
    # Calculate centrality metrics
    analyzer.calculate_centrality_metrics(include_expensive=True)
    print("  ✓ Calculated centrality metrics")
    
    # Calculate community leadership
    analyzer.calculate_community_leadership()
    print("  ✓ Calculated community leadership scores")
    
    # Calculate engagement scores
    comments_df = pd.DataFrame(all_comments)
    if 'author_id' in comments_df.columns:
        analyzer.calculate_engagement_scores(comments_df)
        print("  ✓ Calculated engagement scores")
    
    # Calculate composite score
    analyzer.calculate_composite_score(include_engagement=True, include_community=True)
    print("  ✓ Calculated composite influence scores")
    
    # Get top influencers
    print("\n[5/5] Identifying top influencers...")
    top_influencers = analyzer.get_top_influencers(n=10)
    
    print("\n" + "=" * 60)
    print("Top 10 Influencers")
    print("=" * 60)
    
    for i, (node, score) in enumerate(top_influencers, 1):
        print(f"\n{i}. User: {node}")
        print(f"   Composite Score: {score:.4f}")
        
        # Show individual metrics
        if 'pagerank' in analyzer.metrics:
            print(f"   PageRank: {analyzer.metrics['pagerank'].get(node, 0):.4f}")
        if 'degree' in analyzer.metrics:
            print(f"   Degree Centrality: {analyzer.metrics['degree'].get(node, 0):.4f}")
        if 'betweenness' in analyzer.metrics:
            print(f"   Betweenness: {analyzer.metrics['betweenness'].get(node, 0):.4f}")
        if analyzer.communities:
            print(f"   Community: {analyzer.communities.get(node, -1)}")
    
    # Export results
    output_file = "influencer_results.csv"
    analyzer.export_results(output_file)
    print(f"\n✓ Results exported to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


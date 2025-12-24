"""
Main script for Social Network Analysis
Supports multiple operations: sentiment analysis, community detection, influencer identification
"""
import os
import json
import argparse
from datetime import datetime
from youtube_client import YouTubeClient
from network_builder import NetworkBuilder
from community_detector import CommunityDetector
from visualizer import NetworkVisualizer
from config import MODULARITY_RESOLUTION
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def run_sentiment_analysis(args):
    """Run sentiment analysis on YouTube video comments"""
    print("=" * 60)
    print("YouTube Sentiment Analysis")
    print("=" * 60)
    
    from src.sentiment import SentimentAnalyzer, TextPreprocessor
    import pandas as pd
    
    # Initialize YouTube client
    print("\n[1/4] Initializing YouTube API client...")
    try:
        youtube = YouTubeClient()
        print("✓ YouTube client initialized")
    except Exception as e:
        print(f"✗ Error initializing YouTube client: {e}")
        return
    
    # Get video ID
    video_id = args.video_id
    if not video_id and args.search:
        print(f"\n[2/4] Searching for videos: '{args.search}'...")
        video_ids = youtube.search_videos(args.search, max_results=1)
        if video_ids:
            video_id = video_ids[0]
        else:
            print("No videos found.")
            return
    
    if not video_id:
        print("Please provide --video-id or --search")
        return
    
    # Collect comments
    print(f"\n[2/4] Collecting comments from video {video_id}...")
    comments = youtube.get_video_comments(video_id, max_comments=args.max_comments)
    print(f"✓ Collected {len(comments)} comments")
    
    if len(comments) == 0:
        print("No comments found.")
        return
    
    # Convert to DataFrame
    comments_df = pd.DataFrame(comments)
    
    # Preprocess text
    print("\n[3/4] Preprocessing comments...")
    preprocessor = TextPreprocessor(remove_urls=True, remove_emojis_flag=False)
    comments_df = preprocessor.preprocess_dataframe(comments_df, 'text', 'text_processed')
    print("✓ Comments preprocessed")
    
    # Analyze sentiment
    print(f"\n[4/4] Analyzing sentiment (method: {args.method})...")
    analyzer = SentimentAnalyzer(method=args.method, preprocess=False)
    comments_df = analyzer.analyze_dataframe(comments_df, 'text_processed', 'sentiment')
    print("✓ Sentiment analysis complete")
    
    # Display results
    print("\n" + "=" * 60)
    print("Sentiment Analysis Results")
    print("=" * 60)
    
    sentiment_counts = comments_df['sentiment_label'].value_counts()
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(comments_df)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    avg_score = comments_df['sentiment_score'].mean()
    print(f"\nAverage Sentiment Score: {avg_score:.4f}")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"sentiment_{video_id}_{timestamp}.csv")
    comments_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")


def run_community_detection(args):
    """Run community detection on YouTube comment networks"""
    print("=" * 60)
    print("YouTube Community Detection")
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
    video_ids = []
    if args.video_ids:
        video_ids = args.video_ids.split(',')
    elif args.search:
        print(f"\n[2/5] Searching for videos: '{args.search}'...")
        video_ids = youtube.search_videos(args.search, max_results=args.max_videos)
        print(f"✓ Found {len(video_ids)} videos")
    else:
        print("Please provide --video-ids or --search")
        return
    
    if not video_ids:
        print("No videos found to analyze.")
        return
    
    # Collect comments
    print(f"\n[2/5] Collecting comments from {len(video_ids)} video(s)...")
    all_comments = []
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"  Processing video {i}/{len(video_ids)}: {video_id}")
        video_info = youtube.get_video_info(video_id)
        if video_info:
            print(f"    Title: {video_info['title']}")
        
        comments = youtube.get_video_comments(video_id, max_comments=args.max_comments)
        all_comments.extend(comments)
        print(f"    Collected {len(comments)} comments")
    
    print(f"\n✓ Total comments collected: {len(all_comments)}")
    
    if len(all_comments) < 10:
        print("\n⚠ Warning: Not enough comments for meaningful community detection.")
        return
    
    # Build network
    print("\n[3/5] Building user-user network...")
    network_builder = NetworkBuilder()
    graph = network_builder.build_user_network_from_comments(all_comments)
    
    # Display network statistics
    stats = network_builder.get_network_stats()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if graph.number_of_nodes() < 3:
        print("\n⚠ Warning: Network too small for community detection.")
        return
    
    # Detect communities
    resolution = args.resolution if args.resolution else MODULARITY_RESOLUTION
    print(f"\n[4/5] Detecting communities (resolution={resolution})...")
    detector = CommunityDetector(graph)
    communities = detector.detect_louvain(resolution=resolution)
    
    # Get community statistics
    comm_stats = detector.get_community_stats()
    print("\nCommunity Statistics:")
    print(f"  Total communities: {comm_stats['total_communities']}")
    print(f"  Average community size: {comm_stats['avg_community_size']:.2f}")
    print(f"  Modularity: {comm_stats['modularity']:.4f}")
    
    # Display top communities
    print("\nTop Communities (by size):")
    sorted_communities = sorted(
        comm_stats['communities'].items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]
    
    for comm_id, nodes in sorted_communities:
        print(f"  Community {comm_id}: {len(nodes)} members")
    
    # Visualize
    print("\n[5/5] Creating visualizations...")
    visualizer = NetworkVisualizer(graph, communities)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Network visualization
    print("  Generating network plot...")
    fig = visualizer.plot_network(figsize=(14, 10), node_size=100)
    network_plot_path = os.path.join(output_dir, f"network_{timestamp}.png")
    visualizer.save_plot(fig, network_plot_path)
    plt.close(fig)
    
    # Community size distribution
    print("  Generating community size distribution...")
    fig = visualizer.plot_community_size_distribution()
    comm_dist_path = os.path.join(output_dir, f"community_distribution_{timestamp}.png")
    visualizer.save_plot(fig, comm_dist_path)
    plt.close(fig)
    
    # Degree distribution
    print("  Generating degree distribution...")
    fig = visualizer.plot_degree_distribution()
    degree_dist_path = os.path.join(output_dir, f"degree_distribution_{timestamp}.png")
    visualizer.save_plot(fig, degree_dist_path)
    plt.close(fig)
    
    # Export results
    print("\nExporting results...")
    
    # Export communities
    comm_export_path = os.path.join(output_dir, f"communities_{timestamp}.csv")
    detector.export_communities(comm_export_path)
    
    # Export network
    network_export_path = os.path.join(output_dir, f"network_{timestamp}.gexf")
    network_builder.save_network(network_export_path, format='gexf')
    
    # Export summary
    summary = {
        'timestamp': timestamp,
        'video_ids': video_ids,
        'total_comments': len(all_comments),
        'network_stats': stats,
        'community_stats': {
            'total_communities': comm_stats['total_communities'],
            'modularity': comm_stats['modularity'],
            'avg_community_size': comm_stats['avg_community_size']
        }
    }
    
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to '{output_dir}/' directory:")
    print(f"  - Network plot: {network_plot_path}")
    print(f"  - Community distribution: {comm_dist_path}")
    print(f"  - Degree distribution: {degree_dist_path}")
    print(f"  - Communities CSV: {comm_export_path}")
    print(f"  - Network GEXF: {network_export_path}")
    print(f"  - Summary JSON: {summary_path}")


def run_influencer_identification(args):
    """Run influencer identification on YouTube comment networks"""
    print("=" * 60)
    print("YouTube Influencer Identification")
    print("=" * 60)
    
    from src.influencer import InfluencerAnalyzer
    import pandas as pd
    
    # Initialize YouTube client
    print("\n[1/5] Initializing YouTube API client...")
    try:
        youtube = YouTubeClient()
        print("✓ YouTube client initialized")
    except Exception as e:
        print(f"✗ Error initializing YouTube client: {e}")
        return
    
    # Get video IDs
    video_ids = []
    if args.video_ids:
        video_ids = args.video_ids.split(',')
    elif args.search:
        print(f"\n[2/5] Searching for videos: '{args.search}'...")
        video_ids = youtube.search_videos(args.search, max_results=args.max_videos)
        print(f"✓ Found {len(video_ids)} videos")
    else:
        print("Please provide --video-ids or --search")
        return
    
    if not video_ids:
        print("No videos found to analyze.")
        return
    
    # Collect comments
    print(f"\n[2/5] Collecting comments from {len(video_ids)} video(s)...")
    all_comments = []
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"  Processing video {i}/{len(video_ids)}: {video_id}")
        comments = youtube.get_video_comments(video_id, max_comments=args.max_comments)
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
    if args.include_community:
        analyzer.detect_communities()
        print(f"  ✓ Detected {len(set(analyzer.communities.values()))} communities")
    
    # Calculate centrality metrics
    analyzer.calculate_centrality_metrics(include_expensive=args.include_expensive)
    print("  ✓ Calculated centrality metrics")
    
    # Calculate community leadership
    if args.include_community:
        analyzer.calculate_community_leadership()
        print("  ✓ Calculated community leadership scores")
    
    # Calculate engagement scores
    if args.include_engagement:
        comments_df = pd.DataFrame(all_comments)
        if 'author_id' in comments_df.columns:
            analyzer.calculate_engagement_scores(comments_df)
            print("  ✓ Calculated engagement scores")
    
    # Calculate composite score
    analyzer.calculate_composite_score(
        include_engagement=args.include_engagement,
        include_community=args.include_community
    )
    print("  ✓ Calculated composite influence scores")
    
    # Get top influencers
    print("\n[5/5] Identifying top influencers...")
    top_influencers = analyzer.get_top_influencers(n=args.top_n)
    
    print("\n" + "=" * 60)
    print(f"Top {args.top_n} Influencers")
    print("=" * 60)
    
    for i, (node, score) in enumerate(top_influencers, 1):
        print(f"\n{i}. User: {node}")
        print(f"   Composite Score: {score:.4f}")
        
        # Show individual metrics
        if 'pagerank' in analyzer.metrics:
            print(f"   PageRank: {analyzer.metrics['pagerank'].get(node, 0):.4f}")
        if 'degree' in analyzer.metrics:
            print(f"   Degree Centrality: {analyzer.metrics['degree'].get(node, 0):.4f}")
        if analyzer.communities:
            print(f"   Community: {analyzer.communities.get(node, -1)}")
    
    # Export results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"influencers_{timestamp}.csv")
    analyzer.export_results(output_file)
    print(f"\n✓ Results exported to {output_file}")


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Social Network Analysis for YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sentiment analysis
  python main.py sentiment --video-id dQw4w9WgXcQ --method vader
  
  # Community detection
  python main.py community --search "python tutorial" --max-videos 5
  
  # Influencer identification
  python main.py influencer --search "python tutorial" --top-n 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis type to perform')
    
    # Sentiment analysis parser
    sentiment_parser = subparsers.add_parser('sentiment', help='Perform sentiment analysis')
    sentiment_parser.add_argument('--video-id', type=str, help='YouTube video ID')
    sentiment_parser.add_argument('--search', type=str, help='Search query to find videos')
    sentiment_parser.add_argument('--method', type=str, default='vader', 
                                 choices=['vader', 'textblob', 'hybrid'],
                                 help='Sentiment analysis method')
    sentiment_parser.add_argument('--max-comments', type=int, default=100,
                                 help='Maximum comments to analyze')
    
    # Community detection parser
    community_parser = subparsers.add_parser('community', help='Detect communities')
    community_parser.add_argument('--video-ids', type=str, 
                                 help='Comma-separated list of video IDs')
    community_parser.add_argument('--search', type=str, 
                                 help='Search query to find videos')
    community_parser.add_argument('--max-videos', type=int, default=5,
                                 help='Maximum videos to analyze')
    community_parser.add_argument('--max-comments', type=int, default=100,
                                 help='Maximum comments per video')
    community_parser.add_argument('--resolution', type=float,
                                 help='Resolution parameter for Louvain algorithm')
    
    # Influencer identification parser
    influencer_parser = subparsers.add_parser('influencer', help='Identify influencers')
    influencer_parser.add_argument('--video-ids', type=str,
                                  help='Comma-separated list of video IDs')
    influencer_parser.add_argument('--search', type=str,
                                  help='Search query to find videos')
    influencer_parser.add_argument('--max-videos', type=int, default=5,
                                 help='Maximum videos to analyze')
    influencer_parser.add_argument('--max-comments', type=int, default=100,
                                 help='Maximum comments per video')
    influencer_parser.add_argument('--top-n', type=int, default=10,
                                 help='Number of top influencers to return')
    influencer_parser.add_argument('--include-expensive', action='store_true',
                                   help='Include expensive metrics (betweenness, closeness)')
    influencer_parser.add_argument('--include-engagement', action='store_true', default=True,
                                   help='Include engagement metrics')
    influencer_parser.add_argument('--include-community', action='store_true', default=True,
                                   help='Include community leadership metrics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'sentiment':
            run_sentiment_analysis(args)
        elif args.command == 'community':
            run_community_detection(args)
        elif args.command == 'influencer':
            run_influencer_identification(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

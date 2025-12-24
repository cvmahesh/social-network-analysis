"""
Main script for YouTube Sentiment Analysis
"""
import pandas as pd
from youtube_client import YouTubeClient
from src.sentiment import SentimentAnalyzer, TextPreprocessor
from src.database import DatabaseManager, create_tables
from config import YOUTUBE_API_KEY


def main():
    """Main execution function"""
    print("=" * 60)
    print("YouTube Sentiment Analysis")
    print("=" * 60)
    
    # Initialize YouTube client
    print("\n[1/4] Initializing YouTube API client...")
    try:
        youtube = YouTubeClient()
        print("✓ YouTube client initialized")
    except Exception as e:
        print(f"✗ Error initializing YouTube client: {e}")
        return
    
    # Get video ID or search query
    video_id = input("\nEnter YouTube video ID (or press Enter to search): ").strip()
    
    if not video_id:
        search_query = input("Enter search query: ").strip()
        if search_query:
            video_ids = youtube.search_videos(search_query, max_results=1)
            if video_ids:
                video_id = video_ids[0]
            else:
                print("No videos found.")
                return
        else:
            print("Please provide a video ID or search query.")
            return
    
    # Collect comments
    print(f"\n[2/4] Collecting comments from video {video_id}...")
    comments = youtube.get_video_comments(video_id, max_comments=100)
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
    print("\n[4/4] Analyzing sentiment...")
    analyzer = SentimentAnalyzer(method='vader', preprocess=False)  # Already preprocessed
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
    
    # Show sample comments
    print("\nSample Comments by Sentiment:")
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_comments = comments_df[comments_df['sentiment_label'] == sentiment]
        if len(sentiment_comments) > 0:
            sample = sentiment_comments.iloc[0]
            print(f"\n{sentiment.capitalize()} (score: {sample['sentiment_score']:.4f}):")
            print(f"  {sample['text'][:100]}...")
    
    # Save results
    output_file = f"sentiment_results_{video_id}.csv"
    comments_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


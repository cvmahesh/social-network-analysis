"""
YouTube Data API Client
Fetches video comments and metadata for community detection
"""
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import YOUTUBE_API_KEY

class YouTubeClient:
    """Client for interacting with YouTube Data API v3"""
    
    def __init__(self, api_key=None):
        """
        Initialize YouTube API client
        
        Args:
            api_key: YouTube API key (defaults to config)
        """
        self.api_key = api_key or YOUTUBE_API_KEY
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.request_count = 0
        self.max_requests_per_minute = 100  # YouTube API limit
        
    def get_video_comments(self, video_id, max_comments=100):
        """
        Fetch comments for a given video
        
        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to fetch
            
        Returns:
            List of comment dictionaries with user info and text
        """
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_comments:
                # Calculate how many comments to fetch in this batch
                remaining = max_comments - len(comments)
                max_results = min(100, remaining)  # API max is 100 per request
                
                # Fetch comment threads
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=max_results,
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                
                response = request.execute()
                self.request_count += 1
                
                # Extract comments
                for item in response.get('items', []):
                    top_level_comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'comment_id': item['snippet']['topLevelComment']['id'],
                        'video_id': video_id,
                        'author': top_level_comment['authorDisplayName'],
                        'author_id': top_level_comment.get('authorChannelId', {}).get('value', ''),
                        'text': top_level_comment['textDisplay'],
                        'published_at': top_level_comment['publishedAt'],
                        'like_count': top_level_comment['likeCount'],
                        'reply_count': item['snippet']['totalReplyCount']
                    }
                    comments.append(comment_data)
                    
                    # Fetch replies if any
                    if 'replies' in item:
                        for reply in item['replies']['comments']:
                            reply_data = {
                                'comment_id': reply['id'],
                                'video_id': video_id,
                                'author': reply['snippet']['authorDisplayName'],
                                'author_id': reply['snippet'].get('authorChannelId', {}).get('value', ''),
                                'text': reply['snippet']['textDisplay'],
                                'published_at': reply['snippet']['publishedAt'],
                                'like_count': reply['snippet']['likeCount'],
                                'reply_count': 0,
                                'parent_id': item['snippet']['topLevelComment']['id']
                            }
                            comments.append(reply_data)
                
                # Check if there are more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                # Rate limiting
                time.sleep(0.1)  # Small delay to avoid hitting rate limits
                
        except HttpError as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            
        return comments[:max_comments]
    
    def get_video_info(self, video_id):
        """
        Get video metadata
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video information
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            self.request_count += 1
            
            if response.get('items'):
                video = response['items'][0]
                return {
                    'video_id': video_id,
                    'title': video['snippet']['title'],
                    'description': video['snippet']['description'],
                    'channel_id': video['snippet']['channelId'],
                    'channel_title': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0))
                }
        except HttpError as e:
            print(f"Error fetching video info for {video_id}: {e}")
            
        return None
    
    def search_videos(self, query, max_results=10):
        """
        Search for videos by query
        
        Args:
            query: Search query string
            max_results: Maximum number of videos to return
            
        Returns:
            List of video IDs
        """
        video_ids = []
        try:
            request = self.youtube.search().list(
                part='id',
                q=query,
                type='video',
                maxResults=min(max_results, 50)  # API max is 50 per request
            )
            response = request.execute()
            self.request_count += 1
            
            for item in response.get('items', []):
                video_ids.append(item['id']['videoId'])
                
        except HttpError as e:
            print(f"Error searching videos: {e}")
            
        return video_ids


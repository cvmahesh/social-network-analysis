"""
Configuration management for YouTube Community Detection
Loads environment variables from .env file
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')

# Database Configuration (Optional)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'youtube_communities')
DB_USER = os.getenv('DB_USER', '')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Community Detection Settings
MIN_COMMENTS_PER_USER = int(os.getenv('MIN_COMMENTS_PER_USER', '2'))
MIN_COMMUNITY_SIZE = int(os.getenv('MIN_COMMUNITY_SIZE', '3'))
MODULARITY_RESOLUTION = float(os.getenv('MODULARITY_RESOLUTION', '1.0'))

# Validate required configuration
if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key_here':
    raise ValueError(
        "Please set YOUTUBE_API_KEY in .env file. "
        "Get your API key from: https://console.cloud.google.com/"
    )


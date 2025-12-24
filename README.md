# Social Network Analysis for YouTube

A comprehensive Python toolkit for analyzing YouTube communities, performing sentiment analysis, detecting communities, and identifying influencers using network analysis techniques.

## Features

- **Sentiment Analysis**: Analyze sentiment of YouTube comments using VADER, TextBlob, or hybrid methods
- **Community Detection**: Detect communities in YouTube comment networks using Louvain algorithm
- **Influencer Identification**: Identify key influencers using multiple centrality metrics and composite scoring
- **Network Visualization**: Generate network plots, community distributions, and degree distributions
- **REST API**: Access functionality via REST API endpoints (optional)
- **Export**: Save results in multiple formats (CSV, GEXF, JSON, PNG)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd social-network-analysis
```

Or download and extract the project files.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- NetworkX for network analysis
- FastAPI for REST API (optional)
- VADER and TextBlob for sentiment analysis
- Matplotlib for visualization
- And more...

### Step 4: Get YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable **YouTube Data API v3**
4. Go to **Credentials** → **Create Credentials** → **API Key**
5. Copy your API key

### Step 5: Configure Environment

Create or edit the `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_actual_api_key_here
```

**Important**: Never commit your `.env` file to version control!

## Usage

### Command-Line Interface

The main script supports three operations via command-line arguments:

#### 1. Sentiment Analysis

Analyze sentiment of comments from a YouTube video:

```bash
# Using video ID
python main.py sentiment --video-id dQw4w9WgXcQ --method vader

# Using search query
python main.py sentiment --search "python tutorial" --method vader --max-comments 200

# Available methods: vader, textblob, hybrid
python main.py sentiment --video-id dQw4w9WgXcQ --method hybrid
```

**Options:**
- `--video-id`: YouTube video ID
- `--search`: Search query to find videos (uses first result)
- `--method`: Sentiment method (`vader`, `textblob`, or `hybrid`)
- `--max-comments`: Maximum comments to analyze (default: 100)

**Output:** Results saved to `results/sentiment_VIDEOID_TIMESTAMP.csv`

#### 2. Community Detection

Detect communities in YouTube comment networks:

```bash
# Using search query
python main.py community --search "python tutorial" --max-videos 5

# Using specific video IDs
python main.py community --video-ids "VIDEO_ID1,VIDEO_ID2,VIDEO_ID3" --max-comments 100

# Custom resolution parameter
python main.py community --search "python tutorial" --resolution 1.5
```

**Options:**
- `--video-ids`: Comma-separated list of video IDs
- `--search`: Search query to find videos
- `--max-videos`: Maximum videos to analyze (default: 5)
- `--max-comments`: Maximum comments per video (default: 100)
- `--resolution`: Resolution parameter for Louvain algorithm (optional)

**Output:** 
- Network visualizations (PNG)
- Community assignments (CSV)
- Network file (GEXF)
- Summary statistics (JSON)

#### 3. Influencer Identification

Identify key influencers in YouTube communities:

```bash
# Basic usage
python main.py influencer --search "python tutorial" --top-n 10

# Include expensive metrics
python main.py influencer --search "python tutorial" --include-expensive --top-n 20

# Using specific videos
python main.py influencer --video-ids "VIDEO_ID1,VIDEO_ID2" --top-n 15
```

**Options:**
- `--video-ids`: Comma-separated list of video IDs
- `--search`: Search query to find videos
- `--max-videos`: Maximum videos to analyze (default: 5)
- `--max-comments`: Maximum comments per video (default: 100)
- `--top-n`: Number of top influencers to return (default: 10)
- `--include-expensive`: Include expensive metrics (betweenness, closeness)
- `--include-engagement`: Include engagement metrics (default: True)
- `--include-community`: Include community leadership (default: True)

**Output:** Results saved to `results/influencers_TIMESTAMP.csv`

### Python API (Direct Usage)

You can also use the modules directly in Python:

```python
# Sentiment Analysis
from src.sentiment import SentimentAnalyzer
analyzer = SentimentAnalyzer(method='vader')
result = analyzer.analyze("This is a great video!")
print(result)

# Community Detection
from youtube_client import YouTubeClient
from network_builder import NetworkBuilder
from community_detector import CommunityDetector

youtube = YouTubeClient()
comments = youtube.get_video_comments("VIDEO_ID", max_comments=100)

builder = NetworkBuilder()
graph = builder.build_user_network_from_comments(comments)

detector = CommunityDetector(graph)
communities = detector.detect_louvain()
stats = detector.get_community_stats()
print(f"Found {stats['total_communities']} communities")

# Influencer Identification
from src.influencer import InfluencerAnalyzer

analyzer = InfluencerAnalyzer(graph)
analyzer.calculate_centrality_metrics()
analyzer.calculate_composite_score()
top_influencers = analyzer.get_top_influencers(n=10)
```

### REST API (Optional)

Start the API server:

```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

Access interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

See [API_README.md](API_README.md) for detailed API documentation.

## Project Structure

```
social-network-analysis/
├── src/
│   ├── api/                    # REST API layer
│   ├── sentiment/              # Sentiment analysis module
│   ├── influencer/             # Influencer identification module
│   ├── community/              # Community analysis module
│   ├── database/               # Database operations
│   ├── utils/                  # Utility functions
│   └── ...
├── main.py                     # Main CLI script
├── api_server.py               # API server entry point
├── youtube_client.py           # YouTube API wrapper
├── network_builder.py          # Network construction
├── community_detector.py       # Community detection
├── visualizer.py              # Visualization
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── README.md                  # This file
├── API_README.md              # API documentation
└── results/                   # Output directory (created automatically)
```

## Configuration

Edit `config.py` or `.env` to adjust:

- `MIN_COMMENTS_PER_USER`: Minimum comments required for a user (default: 2)
- `MIN_COMMUNITY_SIZE`: Minimum community size to report (default: 3)
- `MODULARITY_RESOLUTION`: Resolution parameter for Louvain (default: 1.0)

## Output Files

Results are saved in the `results/` directory with timestamps:

### Sentiment Analysis
- `sentiment_VIDEOID_TIMESTAMP.csv`: Comments with sentiment scores

### Community Detection
- `network_TIMESTAMP.png`: Network visualization with communities
- `community_distribution_TIMESTAMP.png`: Community size histogram
- `degree_distribution_TIMESTAMP.png`: Network degree distribution
- `communities_TIMESTAMP.csv`: Community assignments
- `network_TIMESTAMP.gexf`: Network file (open in Gephi)
- `summary_TIMESTAMP.json`: Analysis summary

### Influencer Identification
- `influencers_TIMESTAMP.csv`: Top influencers with all metrics

## Examples

### Example 1: Analyze Sentiment of a Video

```bash
python main.py sentiment --video-id dQw4w9WgXcQ --method vader --max-comments 200
```

### Example 2: Detect Communities in Python Tutorial Videos

```bash
python main.py community --search "python tutorial" --max-videos 10 --max-comments 150
```

### Example 3: Find Top Influencers

```bash
python main.py influencer --search "machine learning" --top-n 20 --include-expensive
```

### Example 4: Complete Analysis Workflow

```bash
# Step 1: Sentiment analysis
python main.py sentiment --search "python tutorial" --method hybrid

# Step 2: Community detection
python main.py community --search "python tutorial" --max-videos 5

# Step 3: Influencer identification
python main.py influencer --search "python tutorial" --top-n 10
```

## API Quota Considerations

YouTube Data API v3 has daily quota limits:
- **Default**: 10,000 units/day
- **Comment threads list**: 1 unit per request
- **Video list**: 1 unit per request
- **Search**: 100 units per request

Monitor your usage in [Google Cloud Console](https://console.cloud.google.com/).

**Tips:**
- Use `--max-comments` to limit API calls
- Cache results when possible
- Consider upgrading quota for production use

## Troubleshooting

### "Please set YOUTUBE_API_KEY in .env file"
- Make sure `.env` file exists in the project root
- Check that the key is not set to the placeholder value
- Verify the API key is valid in Google Cloud Console

### "Not enough comments for meaningful analysis"
- Try videos with more comments
- Increase `--max-comments` parameter
- Analyze multiple videos using `--max-videos`

### "Network too small for analysis"
- Need at least 3 nodes (users) in the network
- Try videos with more diverse commenters
- Lower `MIN_COMMENTS_PER_USER` threshold in config

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one
- Check Python version (3.8+ required)

### API Server Not Starting
- Check if port 8000 is already in use
- Verify FastAPI and uvicorn are installed
- Check for syntax errors in API files

## Advanced Usage

### Custom Analysis Scripts

Create your own scripts using the modules:

```python
from youtube_client import YouTubeClient
from src.sentiment import SentimentAnalyzer
from src.influencer import InfluencerAnalyzer

# Your custom analysis logic here
```

### Batch Processing

Process multiple videos:

```bash
# Create a script to process multiple videos
for video_id in VIDEO_ID_LIST; do
    python main.py sentiment --video-id $video_id
done
```

### Integration with Other Tools

Results can be imported into:
- **Gephi**: Use `.gexf` files for network visualization
- **Excel/Google Sheets**: Import CSV files
- **Python/Pandas**: Load CSV/JSON files directly

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes.

## References

- [YouTube Data API v3 Documentation](https://developers.google.com/youtube/v3)
- [NetworkX Documentation](https://networkx.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [python-louvain Documentation](https://github.com/taynaud/python-louvain)

## Support

For issues, questions, or contributions:
- Check existing issues in the repository
- Create a new issue with detailed information
- Include error messages and steps to reproduce

---

**Happy Analyzing!** 🎉

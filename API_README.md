# Social Network Analysis API

REST API for YouTube community analysis, sentiment analysis, and influencer identification.

## Architecture

The API layer is **completely separate** from the core functionality, allowing you to:
- Use core modules directly via Python (as before)
- Access functionality via REST API endpoints
- Mix and match as needed

```
┌─────────────────────────────────────────┐
│         API Layer (FastAPI)             │
│  - REST endpoints                       │
│  - Request/Response validation          │
│  - Error handling                       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Core Modules (Unchanged)          │
│  - sentiment/                           │
│  - influencer/                          │
│  - community/                           │
│  - utils/                               │
└─────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Make sure your `.env` file has your YouTube API key:
```env
YOUTUBE_API_KEY=your_api_key_here
```

### 3. Start the API Server

```bash
# Option 1: Using the provided script
python api_server.py

# Option 2: Using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Sentiment Analysis

**POST** `/sentiment/analyze`

Analyze sentiment of text(s) or YouTube video comments.

**Request Body:**
```json
{
  "text": "This is a great video!",
  "method": "vader",
  "preprocess": true
}
```

**Or analyze video comments:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "method": "vader",
  "max_comments": 100
}
```

**Response:**
```json
{
  "success": true,
  "method": "vader",
  "results": [
    {
      "text": "This is a great video!",
      "sentiment": "positive",
      "score": 0.6249,
      "positive": 0.5,
      "negative": 0.0,
      "neutral": 0.5
    }
  ],
  "summary": {
    "total_texts": 1,
    "sentiment_distribution": {"positive": 1},
    "average_score": 0.6249
  }
}
```

### Community Detection

**POST** `/community/detect`

Detect communities in YouTube comment networks.

**Request Body:**
```json
{
  "search_query": "python tutorial",
  "max_videos": 5,
  "max_comments_per_video": 100,
  "resolution": 1.0
}
```

**Response:**
```json
{
  "success": true,
  "num_communities": 3,
  "num_nodes": 150,
  "num_edges": 450,
  "modularity": 0.65,
  "communities": [
    {
      "community_id": 0,
      "size": 50,
      "members": ["user1", "user2", ...]
    }
  ]
}
```

### Influencer Identification

**POST** `/influencer/identify`

Identify key influencers in YouTube communities.

**Request Body:**
```json
{
  "search_query": "python tutorial",
  "max_videos": 5,
  "top_n": 10,
  "include_engagement": true,
  "include_community": true
}
```

**Response:**
```json
{
  "success": true,
  "top_influencers": [
    {
      "node_id": "user123",
      "composite_score": 0.85,
      "degree_centrality": 0.45,
      "pagerank": 0.003,
      "engagement_score": 0.92
    }
  ],
  "total_nodes": 150,
  "total_edges": 450
}
```

### Complete Video Analysis

**POST** `/video/analyze`

Complete analysis including sentiment, communities, and influencers.

**Request Body:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "include_sentiment": true,
  "include_communities": true,
  "include_influencers": true
}
```

## Usage Examples

### Using cURL

```bash
# Sentiment analysis
curl -X POST "http://localhost:8000/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is amazing!",
    "method": "vader"
  }'

# Community detection
curl -X POST "http://localhost:8000/community/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "search_query": "python tutorial",
    "max_videos": 5
  }'
```

### Using Python Requests

```python
import requests

# Sentiment analysis
response = requests.post(
    "http://localhost:8000/sentiment/analyze",
    json={
        "text": "This is amazing!",
        "method": "vader"
    }
)
result = response.json()
print(result)

# Influencer identification
response = requests.post(
    "http://localhost:8000/influencer/identify",
    json={
        "search_query": "python tutorial",
        "top_n": 10
    }
)
influencers = response.json()
print(influencers)
```

### Using Direct Python (Still Works!)

The core modules can still be used directly:

```python
from src.sentiment import SentimentAnalyzer
from src.influencer import InfluencerAnalyzer

# Direct usage - no API needed
analyzer = SentimentAnalyzer(method='vader')
result = analyzer.analyze("This is great!")
print(result)
```

## API Structure

```
src/api/
├── __init__.py
├── main.py              # FastAPI app
├── models/
│   ├── __init__.py
│   └── schemas.py      # Pydantic models
└── routers/
    ├── __init__.py
    ├── sentiment.py    # Sentiment endpoints
    ├── community.py    # Community endpoints
    ├── influencer.py   # Influencer endpoints
    └── video.py        # Complete analysis endpoints
```

## Configuration

### CORS

By default, CORS is enabled for all origins. For production, update `src/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Port and Host

Change in `api_server.py`:
```python
uvicorn.run(
    app,
    host="0.0.0.0",  # Listen on all interfaces
    port=8000,       # Port number
    reload=True      # Auto-reload (disable in production)
)
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (video not found)
- `500`: Internal Server Error

Error responses include a `detail` field with error message:
```json
{
  "detail": "Must provide either 'text', 'texts', or 'video_id'"
}
```

## Health Checks

Each router has a health check endpoint:
- `/sentiment/health`
- `/community/health`
- `/influencer/health`
- `/video/health`
- `/health` (main app)

## Production Deployment

### Using Gunicorn + Uvicorn Workers

```bash
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Set these in production:
- `YOUTUBE_API_KEY`: Your YouTube API key
- `DB_HOST`, `DB_PORT`, etc.: Database configuration (if using)

## Testing the API

### Using Swagger UI

1. Start the server: `python api_server.py`
2. Open http://localhost:8000/docs
3. Try endpoints directly from the browser

### Using pytest (if you add tests)

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_sentiment_analysis():
    response = client.post(
        "/sentiment/analyze",
        json={"text": "This is great!", "method": "vader"}
    )
    assert response.status_code == 200
    assert response.json()["success"] == True
```

## Notes

- The API layer is **completely optional** - core modules work independently
- All core functionality remains unchanged
- API adds a REST interface without modifying existing code
- Easy to extend with new endpoints
- Automatic API documentation via Swagger/ReDoc


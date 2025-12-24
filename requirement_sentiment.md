# YouTube Sentiment Analysis - Requirements & Architecture

## Project Overview
This project implements sentiment analysis on YouTube video comments and metadata using various NLP techniques and the YouTube Data API.

---

## Recommended Technology Stack

### 1. Data Collection
- **YouTube Data API v3** - Official API for fetching videos, comments, and metadata
- **Python Libraries**:
  - `google-api-python-client` - Official Google API client
  - `yt-dlp` (optional) - Alternative for video metadata extraction

### 2. Programming Language & Core Libraries
- **Python 3.8+** (recommended: Python 3.10+)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Requests** - HTTP library for API calls

### 3. Sentiment Analysis Options

#### Option A: Pre-trained Models (Fast, Good for General Use)
- **VADER** (`vaderSentiment`) - Rule-based sentiment analyzer, excellent for social media text
- **TextBlob** - Simple API with moderate accuracy, good for quick prototyping

#### Option B: Transformer Models (Higher Accuracy)
- **Hugging Face Transformers** - State-of-the-art transformer models
  - `distilbert-base-uncased-finetuned-sst-2-english` - Fast and accurate
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` - Optimized for social media
- **spaCy** - Advanced NLP library with sentiment models
- **torch** / **tensorflow** - Deep learning frameworks (required for transformers)

#### Option C: Cloud APIs (Managed, Scalable)
- **Google Cloud Natural Language API** - Managed sentiment analysis service
- **AWS Comprehend** - Amazon's NLP service
- **Azure Text Analytics** - Microsoft's text analysis service

### 4. Data Storage
- **SQLite** - Lightweight database for development and small-scale projects
- **PostgreSQL** - Production-grade relational database
- **MongoDB** - NoSQL database for flexible JSON storage
- **CSV/JSON files** - Simple file-based storage for prototyping

### 5. Additional Tools & Libraries
- **Jupyter Notebooks** - Interactive development and exploration
- **FastAPI** / **Flask** - Web framework for REST API (optional)
- **Streamlit** / **Dash** - Dashboard and visualization framework (optional)
- **Docker** - Containerization for deployment
- **python-dotenv** - Environment variable management
- **tqdm** - Progress bars for long-running operations

### 6. Visualization & Analysis
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive visualizations
- **WordCloud** - Word cloud generation

---

## System Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   1. DATA COLLECTION                        │
│  ┌──────────────┐                                           │
│  │ YouTube API  │ → Fetch video metadata, comments, replies │
│  └──────────────┘                                           │
│  • Video details (title, description, view count)           │
│  • Comments (with pagination)                               │
│  • Comment replies                                          │
│  • Rate limit handling                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   2. DATA PREPROCESSING                     │
│  • Clean text (remove URLs, emojis, special chars)          │
│  • Handle encoding issues                                   │
│  • Remove duplicates                                        │
│  • Language detection (optional)                            │
│  • Text normalization                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   3. SENTIMENT ANALYSIS                     │
│  • Apply sentiment model to each comment                    │
│  • Extract: polarity (positive/negative/neutral)            │
│  • Optional: confidence scores                              │
│  • Batch processing for efficiency                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   4. DATA STORAGE                           │
│  • Store raw data + sentiment scores                        │
│  • Aggregate by video/channel                               │
│  • Timestamp tracking                                       │
│  • Indexing for fast queries                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   5. ANALYSIS & VISUALIZATION               │
│  • Sentiment distribution (pie/bar charts)                  │
│  • Trends over time                                         │
│  • Top positive/negative comments                           │
│  • Channel/video comparison                                 │
│  • Export reports (CSV, JSON, PDF)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Setup & Authentication
- [ ] Set up Google Cloud Console project
- [ ] Enable YouTube Data API v3
- [ ] Generate API key or OAuth 2.0 credentials
- [ ] Install Python dependencies
- [ ] Configure environment variables
- [ ] Set up project directory structure

### Phase 2: Data Collection Module
- [ ] Implement YouTube API client
- [ ] Fetch video details (title, description, view count, etc.)
- [ ] Fetch comments with pagination support
- [ ] Fetch comment replies (nested comments)
- [ ] Implement rate limiting and quota management
- [ ] Error handling and retry logic
- [ ] Data validation

### Phase 3: Preprocessing Pipeline
- [ ] Text cleaning (remove HTML, URLs, extra whitespace)
- [ ] Emoji handling (convert to text or remove)
- [ ] Language detection and filtering (optional)
- [ ] Deduplication logic
- [ ] Text normalization
- [ ] Handle special characters and encoding

### Phase 4: Sentiment Analysis Module
- [ ] Select and load sentiment model
- [ ] Process comments in batches for efficiency
- [ ] Extract sentiment labels (positive/negative/neutral)
- [ ] Extract confidence scores
- [ ] Handle edge cases (empty text, very short comments)
- [ ] Model performance evaluation

### Phase 5: Storage & Aggregation
- [ ] Design database schema
- [ ] Implement data storage layer
- [ ] Calculate aggregate metrics:
  - Overall sentiment distribution
  - Sentiment by video
  - Sentiment trends over time
  - Most polarizing comments
- [ ] Implement data retrieval queries

### Phase 6: Reporting & Visualization
- [ ] Generate summary statistics
- [ ] Create visualizations (charts, graphs)
- [ ] Export results (CSV, JSON, reports)
- [ ] Build dashboard (optional)
- [ ] Create REST API endpoints (optional)

---

## Quick Start Recommendations

### Beginner-Friendly Stack
- **Python 3.10+**
- **VADER Sentiment Analyzer** (`vaderSentiment`)
- **YouTube Data API v3** (`google-api-python-client`)
- **SQLite** (via `sqlite3`)
- **Pandas** for data manipulation
- **Matplotlib/Seaborn** for visualization

### Production-Ready Stack
- **Python 3.10+**
- **Hugging Face Transformers** (RoBERTa model)
- **PostgreSQL** with `psycopg2` or `SQLAlchemy`
- **FastAPI** for REST API
- **Docker** for containerization
- **Plotly** for interactive visualizations
- **Redis** (optional) for caching

---

## Key Dependencies

### Core Dependencies
```
google-api-python-client>=2.100.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

### Sentiment Analysis (Choose one or more)
```
# Option A: VADER
vaderSentiment>=3.3.2
textblob>=0.17.1

# Option B: Transformers
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99

# Option C: spaCy
spacy>=3.5.0
```

### Data Storage
```
# SQLite (built-in, no install needed)
# PostgreSQL
psycopg2-binary>=2.9.6
sqlalchemy>=2.0.0

# MongoDB
pymongo>=4.4.0
```

### Visualization
```
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
wordcloud>=1.9.2
```

### Optional: Web Framework
```
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.24.0
```

### Development Tools
```
jupyter>=1.0.0
ipython>=8.12.0
pytest>=7.4.0
black>=23.7.0
flake8>=6.1.0
```

---

## Environment Variables

Create a `.env` file with the following:

```env
# YouTube API Configuration
YOUTUBE_API_KEY=your_api_key_here

# Database Configuration (if using PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=youtube_sentiment
DB_USER=your_username
DB_PASSWORD=your_password

# Optional: Cloud API Keys
GOOGLE_CLOUD_API_KEY=your_google_cloud_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

---

## API Quota Considerations

- **YouTube Data API v3** has daily quota limits (default: 10,000 units/day)
- **Cost per operation**:
  - Video list: 1 unit
  - Comment threads list: 1 unit
  - Comment replies: 1 unit
- **Best practices**:
  - Implement caching to reduce API calls
  - Batch requests when possible
  - Monitor quota usage
  - Consider upgrading quota for production use

---

## Project Structure Recommendation

```
social-network-analysis/
├── requirement_sentiment.md  # This file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (gitignored)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── youtube_client.py    # YouTube API wrapper
│   ├── preprocessor.py      # Text preprocessing
│   ├── sentiment_analyzer.py # Sentiment analysis
│   ├── database.py          # Database operations
│   └── visualizer.py        # Visualization functions
├── data/
│   ├── raw/                 # Raw data from API
│   ├── processed/           # Processed data
│   └── results/             # Analysis results
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
└── main.py                  # Main execution script
```

---

## Next Steps

1. Choose your sentiment analysis approach (VADER, Transformers, or Cloud API)
2. Set up YouTube API credentials
3. Install dependencies from `requirements.txt`
4. Implement data collection module
5. Build preprocessing pipeline
6. Integrate sentiment analysis
7. Set up data storage
8. Create visualization and reporting

---

## References

- [YouTube Data API v3 Documentation](https://developers.google.com/youtube/v3)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs)


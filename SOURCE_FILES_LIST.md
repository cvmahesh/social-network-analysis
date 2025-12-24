# Required Source Files List

This document lists all required Python source files needed to implement the features described in the requirement documents.

---

## File Status Legend
- ✅ **Exists**: File already created
- ❌ **Missing**: File needs to be created
- 🔄 **Needs Update**: File exists but needs enhancements

---

## Core Infrastructure Files

### Configuration & Setup
- ✅ `config.py` - Configuration management, environment variables
- ❌ `src/__init__.py` - Package initialization
- ✅ `.env` - Environment variables (with placeholders)
- ✅ `.gitignore` - Git ignore rules

---

## Data Collection & API

### YouTube API Integration
- ✅ `youtube_client.py` - YouTube Data API v3 wrapper
  - Fetch videos, comments, channels
  - Handle pagination and rate limiting
  - Error handling and retry logic

**Enhancements Needed**:
- Add subscription data fetching
- Add channel metadata collection
- Add video search with filters

---

## Sentiment Analysis Module

### Core Sentiment Analysis Files
- ❌ `src/sentiment/sentiment_analyzer.py` - Main sentiment analysis class
  - VADER sentiment analyzer integration
  - Transformer model support (Hugging Face)
  - Cloud API integration (optional)
  - Batch processing support

- ❌ `src/sentiment/preprocessor.py` - Text preprocessing
  - Text cleaning (URLs, HTML, special chars)
  - Emoji handling
  - Language detection
  - Text normalization
  - Deduplication

- ❌ `src/sentiment/sentiment_models.py` - Model management
  - Model loading and caching
  - Model selection logic
  - Model performance evaluation

- ❌ `src/sentiment/__init__.py` - Sentiment module initialization

**Reference**: `requirement_sentiment.md`

---

## Community Detection Module

### Network Construction
- ✅ `network_builder.py` - Network construction from YouTube data
  - User-user network building
  - Reply network construction
  - Edge weight calculation
  - Network statistics

**Enhancements Needed**:
- Channel-channel network building
- Video-video network construction
- Bipartite network support
- Temporal network building

### Community Detection
- ✅ `community_detector.py` - Community detection algorithms
  - Louvain algorithm implementation
  - Community statistics
  - Community export

**Enhancements Needed**:
- Leiden algorithm support
- Label Propagation Algorithm (LPA)
- Dynamic community detection
- Overlapping community detection

- ❌ `src/community/community_analyzer.py` - Community analysis
  - Community lifecycle tracking
  - Community merging/splitting detection
  - Community topic evolution
  - Cross-community analysis

- ❌ `src/community/network_metrics.py` - Network metrics calculation
  - Centrality calculations wrapper
  - Network topology metrics
  - Community-specific metrics

- ❌ `src/community/__init__.py` - Community module initialization

**Reference**: `requirement_community_detection.md`

---

## Influencer Identification Module

### Core Influencer Analysis
- ❌ `src/influencer/influencer_analyzer.py` - Main influencer identification system
  - Composite influence score calculation
  - Multi-metric combination
  - Influencer ranking
  - Complete workflow implementation

- ❌ `src/influencer/centrality_calculator.py` - Centrality metrics calculation
  - Degree centrality
  - Betweenness centrality
  - Closeness centrality
  - Eigenvector centrality / PageRank
  - Clustering coefficient
  - Performance optimizations for large networks

- ❌ `src/influencer/engagement_calculator.py` - Engagement metrics
  - Comment engagement score
  - Reply influence calculation
  - Early commenter score
  - Engagement velocity

- ❌ `src/influencer/community_metrics.py` - Community-based metrics
  - Community leadership score
  - Cross-community influence
  - Community bridge score
  - Within-community vs. global influence

- ❌ `src/influencer/composite_scorer.py` - Composite scoring system
  - Metric normalization
  - Weighted combination
  - Custom weight configuration
  - Score ranking and export

- ❌ `src/influencer/temporal_metrics.py` - Temporal influence metrics
  - Activity pattern analysis
  - Growth trajectory calculation
  - Sustained influence measurement
  - Influence evolution tracking

- ❌ `src/influencer/__init__.py` - Influencer module initialization

**Reference**: `requirement_influencer_identification.md`

---

## Historical Analysis Module

### Temporal Analysis
- ❌ `src/historical/temporal_analyzer.py` - Main temporal analysis class
  - Time-series data processing
  - Snapshot comparison
  - Temporal network analysis
  - Historical data reconstruction

- ❌ `src/historical/community_evolution.py` - Community evolution tracking
  - Community lifecycle detection
  - Community merging/splitting
  - Community topic evolution
  - Community growth patterns

- ❌ `src/historical/network_evolution.py` - Network structure evolution
  - Network growth tracking
  - Centrality evolution
  - Topology changes detection
  - Structural transition identification

- ❌ `src/historical/user_evolution.py` - User behavior evolution
  - Activity pattern analysis
  - Role evolution tracking
  - Influence trajectory calculation
  - User cohort analysis

- ❌ `src/historical/time_series_analyzer.py` - Time-series analysis
  - Trend analysis
  - Seasonality detection
  - Change point detection
  - Forecasting models

- ❌ `src/historical/event_detector.py` - Event and anomaly detection
  - Spike detection
  - Anomaly identification
  - Event impact analysis
  - Pattern recognition

- ❌ `src/historical/data_storage.py` - Temporal data storage
  - Snapshot management
  - Incremental updates
  - Time-series database integration
  - Historical data retrieval

- ❌ `src/historical/__init__.py` - Historical module initialization

**Reference**: `requirement_historical_analysis.md`

---

## Data Storage & Management

### Database Operations
- ❌ `src/database/database.py` - Database operations
  - SQLite/PostgreSQL integration
  - Data insertion and retrieval
  - Query optimization
  - Connection management

- ❌ `src/database/schema.py` - Database schema definitions
  - Table creation scripts
  - Index definitions
  - Migration scripts

- ❌ `src/database/time_series_db.py` - Time-series database operations
  - InfluxDB/TimescaleDB integration
  - Time-series data storage
  - Temporal queries
  - Data aggregation

- ❌ `src/database/graph_db.py` - Graph database operations (optional)
  - Neo4j integration
  - Graph queries
  - Network storage
  - Graph operations

- ❌ `src/database/__init__.py` - Database module initialization

---

## Visualization & Reporting

### Visualization
- ✅ `visualizer.py` - Network visualization
  - Network plots with communities
  - Community size distribution
  - Degree distribution

**Enhancements Needed**:
- Temporal visualization
- Interactive plots (Plotly)
- Network animations
- Trajectory visualizations
- Sankey diagrams

- ❌ `src/visualization/temporal_visualizer.py` - Temporal visualizations
  - Time-series plots
  - Network evolution animations
  - Trajectory plots
  - Stream graphs

- ❌ `src/visualization/influencer_visualizer.py` - Influencer visualizations
  - Influence score rankings
  - Centrality comparisons
  - Influence trajectory plots
  - Community leadership visualizations

- ❌ `src/visualization/report_generator.py` - Report generation
  - PDF report generation
  - HTML dashboards
  - Summary statistics
  - Export functionality

- ❌ `src/visualization/__init__.py` - Visualization module initialization

---

## Utilities & Helpers

### Utility Functions
- ❌ `src/utils/text_utils.py` - Text processing utilities
  - Text cleaning functions
  - Emoji handling
  - Language detection
  - Text normalization

- ❌ `src/utils/network_utils.py` - Network utility functions
  - Graph conversion utilities
  - Network validation
  - Graph statistics
  - Network comparison

- ❌ `src/utils/data_utils.py` - Data manipulation utilities
  - Data cleaning
  - Data transformation
  - Data validation
  - Data export/import

- ❌ `src/utils/time_utils.py` - Time-related utilities
  - Timestamp handling
  - Time period calculations
  - Temporal alignment
  - Time zone handling

- ❌ `src/utils/metrics_utils.py` - Metrics calculation utilities
  - Metric normalization
  - Statistical functions
  - Aggregation functions
  - Comparison utilities

- ❌ `src/utils/__init__.py` - Utils module initialization

---

## Feature Extraction & NLP

### NLP & Feature Extraction
- ❌ `src/features/feature_extractor.py` - Feature extraction
  - TF-IDF extraction
  - Text embeddings
  - Topic modeling
  - Keyword extraction

- ❌ `src/features/topic_modeler.py` - Topic modeling
  - LDA implementation
  - BERTopic integration
  - Topic evolution tracking
  - Topic visualization

- ❌ `src/features/embeddings.py` - Text embeddings
  - Sentence transformer integration
  - Word2Vec/Gensim
  - Embedding similarity
  - Semantic analysis

- ❌ `src/features/__init__.py` - Features module initialization

---

## Main Execution Scripts

### Entry Points
- ✅ `main.py` - Main execution script (community detection)
  - Complete workflow demonstration
  - Command-line interface

**Additional Main Scripts Needed**:
- ❌ `main_sentiment.py` - Sentiment analysis main script
- ❌ `main_influencer.py` - Influencer identification main script
- ❌ `main_historical.py` - Historical analysis main script
- ❌ `main_complete.py` - Complete analysis pipeline

---

## Testing

### Test Files
- ❌ `tests/__init__.py` - Test package initialization
- ❌ `tests/test_youtube_client.py` - YouTube client tests
- ❌ `tests/test_network_builder.py` - Network builder tests
- ❌ `tests/test_community_detector.py` - Community detection tests
- ❌ `tests/test_sentiment_analyzer.py` - Sentiment analysis tests
- ❌ `tests/test_influencer_analyzer.py` - Influencer identification tests
- ❌ `tests/test_temporal_analyzer.py` - Historical analysis tests
- ❌ `tests/test_utils.py` - Utility functions tests
- ❌ `tests/test_integration.py` - Integration tests
- ❌ `tests/fixtures/` - Test fixtures and sample data

---

## Configuration & Documentation

### Configuration Files
- ✅ `requirements.txt` - Python dependencies
- ❌ `setup.py` - Package setup script
- ❌ `pyproject.toml` - Modern Python project configuration
- ❌ `Dockerfile` - Docker containerization (optional)
- ❌ `docker-compose.yml` - Docker Compose configuration (optional)

### Documentation
- ✅ `README.md` - Project documentation
- ✅ `requirement_sentiment.md` - Sentiment analysis requirements
- ✅ `requirement_community_detection.md` - Community detection requirements
- ✅ `requirement_influencer_identification.md` - Influencer identification requirements
- ✅ `requirement_historical_analysis.md` - Historical analysis requirements
- ❌ `API_DOCUMENTATION.md` - API documentation
- ❌ `CONTRIBUTING.md` - Contribution guidelines

---

## Project Structure Summary

```
social-network-analysis/
├── src/
│   ├── __init__.py
│   ├── config.py                    ✅
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py
│   │   ├── preprocessor.py
│   │   └── sentiment_models.py
│   ├── community/
│   │   ├── __init__.py
│   │   ├── community_analyzer.py
│   │   └── network_metrics.py
│   ├── influencer/
│   │   ├── __init__.py
│   │   ├── influencer_analyzer.py
│   │   ├── centrality_calculator.py
│   │   ├── engagement_calculator.py
│   │   ├── community_metrics.py
│   │   ├── composite_scorer.py
│   │   └── temporal_metrics.py
│   ├── historical/
│   │   ├── __init__.py
│   │   ├── temporal_analyzer.py
│   │   ├── community_evolution.py
│   │   ├── network_evolution.py
│   │   ├── user_evolution.py
│   │   ├── time_series_analyzer.py
│   │   ├── event_detector.py
│   │   └── data_storage.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── schema.py
│   │   ├── time_series_db.py
│   │   └── graph_db.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── temporal_visualizer.py
│   │   ├── influencer_visualizer.py
│   │   └── report_generator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   ├── topic_modeler.py
│   │   └── embeddings.py
│   └── utils/
│       ├── __init__.py
│       ├── text_utils.py
│       ├── network_utils.py
│       ├── data_utils.py
│       ├── time_utils.py
│       └── metrics_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_youtube_client.py
│   ├── test_network_builder.py
│   ├── test_community_detector.py
│   ├── test_sentiment_analyzer.py
│   ├── test_influencer_analyzer.py
│   ├── test_temporal_analyzer.py
│   ├── test_utils.py
│   ├── test_integration.py
│   └── fixtures/
├── notebooks/                        # Jupyter notebooks for exploration
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── main.py                           ✅
├── main_sentiment.py
├── main_influencer.py
├── main_historical.py
├── main_complete.py
├── youtube_client.py                 ✅
├── network_builder.py                ✅
├── community_detector.py             ✅
├── visualizer.py                     ✅
├── config.py                         ✅
├── requirements.txt                  ✅
├── README.md                          ✅
├── requirement_sentiment.md           ✅
├── requirement_community_detection.md ✅
├── requirement_influencer_identification.md ✅
├── requirement_historical_analysis.md ✅
└── SOURCE_FILES_LIST.md              ✅ (this file)
```

---

## Priority Implementation Order

### Phase 1: Core Infrastructure (High Priority)
1. `src/__init__.py`
2. `src/utils/` - All utility modules
3. `src/database/database.py` - Basic database operations
4. Enhance existing files with missing features

### Phase 2: Sentiment Analysis (High Priority)
1. `src/sentiment/preprocessor.py`
2. `src/sentiment/sentiment_analyzer.py`
3. `main_sentiment.py`

### Phase 3: Community Detection Enhancements (Medium Priority)
1. `src/community/community_analyzer.py`
2. `src/community/network_metrics.py`
3. Enhance `community_detector.py` with additional algorithms

### Phase 4: Influencer Identification (High Priority)
1. `src/influencer/centrality_calculator.py`
2. `src/influencer/engagement_calculator.py`
3. `src/influencer/composite_scorer.py`
4. `src/influencer/influencer_analyzer.py`
5. `main_influencer.py`

### Phase 5: Historical Analysis (Medium Priority)
1. `src/historical/temporal_analyzer.py`
2. `src/historical/community_evolution.py`
3. `src/historical/time_series_analyzer.py`
4. `main_historical.py`

### Phase 6: Advanced Features (Low Priority)
1. `src/features/` - Feature extraction modules
2. `src/visualization/` - Advanced visualizations
3. `src/database/graph_db.py` - Graph database support
4. `main_complete.py` - Complete pipeline

### Phase 7: Testing & Documentation (Ongoing)
1. All test files
2. API documentation
3. Additional documentation

---

## File Count Summary

- **Total Files**: ~60+ Python source files
- **Existing Files**: 6 files ✅
- **Files to Create**: ~54 files ❌
- **Files Needing Updates**: 4 files 🔄

---

## Notes

1. **Modular Design**: Files are organized into logical modules for maintainability
2. **Incremental Implementation**: Can be implemented in phases
3. **Optional Components**: Some files (like graph_db.py) are optional depending on requirements
4. **Testing**: Each module should have corresponding test files
5. **Documentation**: Each module should have docstrings and type hints

---

## Quick Reference

To see which files are needed for a specific feature:
- **Sentiment Analysis**: See `requirement_sentiment.md` → Files in `src/sentiment/`
- **Community Detection**: See `requirement_community_detection.md` → Files in `src/community/`
- **Influencer Identification**: See `requirement_influencer_identification.md` → Files in `src/influencer/`
- **Historical Analysis**: See `requirement_historical_analysis.md` → Files in `src/historical/`


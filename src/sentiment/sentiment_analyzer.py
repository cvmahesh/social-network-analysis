"""
Sentiment analysis using various methods
"""
from typing import Dict, List, Optional, Union
import pandas as pd
from .preprocessor import TextPreprocessor

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class SentimentAnalyzer:
    """Sentiment analysis using multiple methods"""
    
    def __init__(self, method: str = 'vader', preprocess: bool = True):
        """
        Initialize sentiment analyzer
        
        Args:
            method: Method to use ('vader', 'textblob', 'hybrid')
            preprocess: Whether to preprocess text before analysis
        """
        self.method = method
        self.preprocessor = TextPreprocessor() if preprocess else None
        
        # Initialize analyzers
        if method in ['vader', 'hybrid'] and VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
        
        if method in ['textblob', 'hybrid'] and TEXTBLOB_AVAILABLE:
            self.textblob_available = True
        else:
            self.textblob_available = False
    
    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # Preprocess if enabled
        if self.preprocessor:
            text = self.preprocessor.preprocess(text)
        
        if not text:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        if self.method == 'vader':
            return self._analyze_vader(text)
        elif self.method == 'textblob':
            return self._analyze_textblob(text)
        elif self.method == 'hybrid':
            return self._analyze_hybrid(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _analyze_vader(self, text: str) -> Dict[str, Union[float, str]]:
        """Analyze using VADER"""
        if not self.vader_analyzer:
            raise ValueError("VADER not available. Install vaderSentiment.")
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'method': 'vader'
        }
    
    def _analyze_textblob(self, text: str) -> Dict[str, Union[float, str]]:
        """Analyze using TextBlob"""
        if not self.textblob_available:
            raise ValueError("TextBlob not available. Install textblob.")
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Determine sentiment label
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': polarity,
            'positive': max(0, polarity),
            'negative': max(0, -polarity),
            'neutral': 1 - abs(polarity),
            'method': 'textblob'
        }
    
    def _analyze_hybrid(self, text: str) -> Dict[str, Union[float, str]]:
        """Analyze using both methods and combine"""
        vader_result = self._analyze_vader(text) if self.vader_analyzer else None
        textblob_result = self._analyze_textblob(text) if self.textblob_available else None
        
        if vader_result and textblob_result:
            # Average the scores
            avg_score = (vader_result['score'] + textblob_result['score']) / 2
            
            if avg_score >= 0.05:
                sentiment = 'positive'
            elif avg_score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': avg_score,
                'vader_score': vader_result['score'],
                'textblob_score': textblob_result['score'],
                'positive': (vader_result['positive'] + textblob_result['positive']) / 2,
                'negative': (vader_result['negative'] + textblob_result['negative']) / 2,
                'neutral': (vader_result['neutral'] + textblob_result['neutral']) / 2,
                'method': 'hybrid'
            }
        elif vader_result:
            return vader_result
        elif textblob_result:
            return textblob_result
        else:
            raise ValueError("No sentiment analysis method available")
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame,
                         text_column: str,
                         output_prefix: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze sentiment of texts in DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_prefix: Prefix for output columns
            
        Returns:
            DataFrame with sentiment analysis columns
        """
        df_result = df.copy()
        
        # Analyze each text
        results = df[text_column].apply(self.analyze)
        
        # Extract results into columns
        df_result[f'{output_prefix}_label'] = results.apply(lambda x: x['sentiment'])
        df_result[f'{output_prefix}_score'] = results.apply(lambda x: x['score'])
        df_result[f'{output_prefix}_positive'] = results.apply(lambda x: x.get('positive', 0))
        df_result[f'{output_prefix}_negative'] = results.apply(lambda x: x.get('negative', 0))
        df_result[f'{output_prefix}_neutral'] = results.apply(lambda x: x.get('neutral', 0))
        
        return df_result


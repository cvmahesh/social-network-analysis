"""
Text preprocessing for sentiment analysis
"""
from typing import List, Dict, Optional
import pandas as pd
from src.utils.text_utils import clean_text, remove_emojis, normalize_text


class TextPreprocessor:
    """Text preprocessing pipeline for sentiment analysis"""
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emojis_flag: bool = False,
                 normalize: bool = True,
                 min_length: int = 1):
        """
        Initialize preprocessor
        
        Args:
            remove_urls: Whether to remove URLs
            remove_emojis_flag: Whether to remove emojis
            normalize: Whether to normalize text
            min_length: Minimum text length after preprocessing
        """
        self.remove_urls = remove_urls
        self.remove_emojis_flag = remove_emojis_flag
        self.normalize = normalize
        self.min_length = min_length
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Clean text
        processed = clean_text(text, 
                              remove_urls=self.remove_urls,
                              remove_emojis_flag=self.remove_emojis_flag)
        
        # Normalize if requested
        if self.normalize:
            processed = normalize_text(processed)
        
        # Check minimum length
        if len(processed) < self.min_length:
            return ""
        
        return processed
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def preprocess_dataframe(self, 
                           df: pd.DataFrame, 
                           text_column: str,
                           output_column: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name of output column (default: text_column + '_processed')
            
        Returns:
            DataFrame with preprocessed text column
        """
        df_processed = df.copy()
        
        if output_column is None:
            output_column = f"{text_column}_processed"
        
        df_processed[output_column] = df[text_column].apply(self.preprocess)
        
        return df_processed
    
    def remove_empty(self, texts: List[str]) -> List[str]:
        """
        Remove empty texts after preprocessing
        
        Args:
            texts: List of texts
            
        Returns:
            List with empty texts removed
        """
        return [text for text in texts if text and len(text.strip()) >= self.min_length]


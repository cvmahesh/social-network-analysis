"""
Text processing utilities for comment analysis
"""
import re
import html
from typing import Optional
import emoji


def clean_text(text: str, remove_urls: bool = True, remove_emojis_flag: bool = False) -> str:
    """
    Clean text by removing URLs, HTML entities, and optionally emojis
    
    Args:
        text: Input text to clean
        remove_urls: Whether to remove URLs
        remove_emojis_flag: Whether to remove emojis
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    if remove_urls:
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        text = url_pattern.sub('', text)
    
    # Remove emojis if requested
    if remove_emojis_flag:
        text = remove_emojis(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_emojis(text: str) -> str:
    """
    Remove emojis from text
    
    Args:
        text: Input text
        
    Returns:
        Text without emojis
    """
    return emoji.replace_emoji(text, replace='')


def extract_emojis(text: str) -> list:
    """
    Extract emojis from text
    
    Args:
        text: Input text
        
    Returns:
        List of emojis found
    """
    return [char for char in text if char in emoji.EMOJI_DATA]


def detect_language(text: str) -> Optional[str]:
    """
    Simple language detection based on character patterns
    For more accurate detection, use langdetect library
    
    Args:
        text: Input text
        
    Returns:
        Language code (e.g., 'en', 'es') or None
    """
    # Simple heuristic: check for common patterns
    # For production, use langdetect or similar library
    
    if not text:
        return None
    
    # Check for English patterns
    english_pattern = re.compile(r'[a-zA-Z]+')
    if english_pattern.search(text):
        return 'en'
    
    # Add more language detection logic here
    return None


def normalize_text(text: str) -> str:
    """
    Normalize text (lowercase, remove special chars)
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_hashtags(text: str) -> list:
    """
    Extract hashtags from text
    
    Args:
        text: Input text
        
    Returns:
        List of hashtags (without #)
    """
    hashtag_pattern = re.compile(r'#(\w+)')
    return hashtag_pattern.findall(text)


def extract_mentions(text: str) -> list:
    """
    Extract mentions from text
    
    Args:
        text: Input text
        
    Returns:
        List of mentions (without @)
    """
    mention_pattern = re.compile(r'@(\w+)')
    return mention_pattern.findall(text)


def calculate_text_statistics(text: str) -> dict:
    """
    Calculate basic text statistics
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'length': 0,
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'length': len(text),
        'word_count': len(words),
        'char_count': len(text.replace(' ', '')),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
    }


"""
Tests for token counting operations
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTokenCounting:
    """Tests for token counting functions."""

    def test_count_tokens_returns_integer(self):
        """Test that count_tokens returns an integer."""
        from main import count_tokens
        
        result = count_tokens("Hello world")
        
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self):
        """Test token count for empty string."""
        from main import count_tokens
        
        result = count_tokens("")
        
        assert isinstance(result, int)
        assert result >= 0

    def test_count_tokens_longer_text(self):
        """Test token count for longer text."""
        from main import count_tokens
        
        text = "This is a longer piece of text that should have more tokens. " * 10
        result = count_tokens(text)
        
        assert isinstance(result, int)
        assert result > 10

    def test_count_tokens_with_special_characters(self):
        """Test token count with special characters."""
        from main import count_tokens
        
        text = "Hello! How are you? I'm fine. \n\nNew line here."
        result = count_tokens(text)
        
        assert isinstance(result, int)
        assert result > 0


class TestTokenCountingFallback:
    """Tests for token counting fallback behavior."""

    def test_fallback_returns_reasonable_estimate(self):
        """Test that fallback returns reasonable estimate."""
        from main import count_tokens
        
        text = "test"
        result = count_tokens(text)
        
        assert isinstance(result, int)
        assert result >= 1

    def test_fallback_scales_with_length(self):
        """Test that fallback scales with text length."""
        from main import count_tokens
        
        short_text = "hi"
        long_text = "hello world this is a test"
        
        short_count = count_tokens(short_text)
        long_count = count_tokens(long_text)
        
        assert long_count > short_count

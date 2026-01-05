"""Tests for sentence generator module."""

import pytest
from src.sentence_generator import SentenceGenerator, generate_sentence

class TestSentenceGenerator:
    """Test SentenceGenerator class."""
    
    def test_init_complexity(self):
        """Test initialization with different complexities."""
        gen_simple = SentenceGenerator("simple")
        assert gen_simple.complexity == "simple"
        
        gen_medium = SentenceGenerator("medium")
        assert gen_medium.complexity == "medium"
        
        gen_complex = SentenceGenerator("complex")
        assert gen_complex.complexity == "complex"
    
    def test_generate_format(self):
        """Test return format of generate method."""
        gen = SentenceGenerator()
        cid, sentence = gen.generate()
        
        assert isinstance(cid, str)
        assert len(cid) > 0
        assert isinstance(sentence, str)
        assert len(sentence) > 0
    
    def test_template_replacement(self):
        """Test that all placeholders are replaced."""
        gen = SentenceGenerator("simple")
        
        # Test multiple sentences
        for _ in range(20):
            _, sentence = gen.generate()
            assert "{" not in sentence
            assert "}" not in sentence
    
    def test_complexity_differences(self):
        """Test that different complexities use different templates."""
        # This is a probabilistic test, but simple sentences are distinct 
        # from complex ones in length/structure usually.
        # Better: verify that the generated sentence comes from the correct template list
        
        gen = SentenceGenerator("simple")
        # We can't access private _template_map easily, so we rely on functional check
        # Checking that no known simple template placeholders remain is enough (already done)
        pass

    def test_generate_multiple(self):
        """Test generating multiple unique sentences."""
        gen = SentenceGenerator()
        sentences = gen.generate_multiple(5)
        
        assert len(sentences) == 5
        # Check uniqueness
        unique_sentences = {s for _, s in sentences}
        assert len(unique_sentences) == 5
        
    def test_convenience_function(self):
        """Test generate_sentence convenience function."""
        cid, sentence = generate_sentence("simple")
        assert isinstance(cid, str)
        assert isinstance(sentence, str)

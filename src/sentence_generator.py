"""Random sentence generator for liveness detection."""

import random
import uuid
from typing import Literal

# Sentence templates organized by complexity
SIMPLE_TEMPLATES = [
    "The {color} {object} is {location}",
    "My favorite {category} is {item}",
    "Today is a {adjective} {time_period}",
    "I like to {activity} on {day}",
    "The number {number} is {adjective}",
    "My {object} is {color}",
    "I can see a {color} {object}",
    "The weather is {adjective} today",
    "I enjoy {activity} very much",
    "This is my {adjective} {object}",
]

MEDIUM_TEMPLATES = [
    "I would like to {activity} at the {location} tomorrow",
    "The {adjective} {object} belongs to my {relation}",
    "Every {day} I {activity} with my {relation}",
    "My {relation} has a {color} {object}",
    "The {number} {color} {object}s are in the {location}",
    "I prefer {activity} over {activity} on weekends",

]

COMPLEX_TEMPLATES = [
    "On {day} I went to the {location} and saw a {color} {object}",
    "My {relation} told me that {activity} is better than {activity}",
    "I believe the {adjective} {object} should be placed in the {location}",
    "The {number} {color} {object}s that I saw were absolutely {adjective}",
]

# Word banks for filling templates
WORD_BANKS = {
    "color": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown"],
    "object": ["car", "book", "phone", "table", "chair", "lamp", "computer", "bag", "pen", "cup"],
    "location": ["outside", "inside", "upstairs", "downstairs", "nearby", "here", "there", "home", "office", "garden"],
    "category": ["color", "number", "food", "animal", "season", "day", "month"],
    "item": ["seven", "blue", "pizza", "cat", "summer", "Friday", "January"],
    "adjective": ["beautiful", "wonderful", "amazing", "terrible", "great", "small", "large", "bright", "dark", "quiet"],
    "time_period": ["day", "morning", "evening", "afternoon", "night", "week", "month", "year"],
    "activity": ["read", "write", "walk", "run", "swim", "cook", "sleep", "work", "play", "study"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "number": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "relation": ["friend", "family", "colleague", "neighbor", "brother", "sister", "parent"],
}


class SentenceGenerator:
    """Generate random sentences for liveness detection."""
    
    def __init__(self, complexity: Literal["simple", "medium", "complex"] = "simple"):
        """
        Initialize sentence generator.
        
        Args:
            complexity: Sentence complexity level
        """
        self.complexity = complexity
        self._template_map = {
            "simple": SIMPLE_TEMPLATES,
            "medium": MEDIUM_TEMPLATES,
            "complex": COMPLEX_TEMPLATES,
        }
    
    def generate(self) -> tuple[str, str]:
        """
        Generate a random sentence.
        
        Returns:
            Tuple of (challenge_id, sentence)
        """
        # Select random template
        templates = self._template_map[self.complexity]
        template = random.choice(templates)
        
        # Fill template with random words
        sentence = template
        for placeholder, words in WORD_BANKS.items():
            if f"{{{placeholder}}}" in sentence:
                # Replace all occurrences of this placeholder
                while f"{{{placeholder}}}" in sentence:
                    word = random.choice(words)
                    sentence = sentence.replace(f"{{{placeholder}}}", word, 1)
        
        # Generate unique challenge ID
        challenge_id = str(uuid.uuid4())
        
        return challenge_id, sentence
    
    def generate_multiple(self, count: int = 5) -> list[tuple[str, str]]:
        """
        Generate multiple unique sentences.
        
        Args:
            count: Number of sentences to generate
            
        Returns:
            List of (challenge_id, sentence) tuples
        """
        sentences = set()
        results = []
        
        # Keep generating until we have enough unique sentences
        while len(results) < count:
            challenge_id, sentence = self.generate()
            if sentence not in sentences:
                sentences.add(sentence)
                results.append((challenge_id, sentence))
        
        return results


def generate_sentence(complexity: Literal["simple", "medium", "complex"] = "medium") -> tuple[str, str]:
    """
    Convenience function to generate a single sentence.
    
    Args:
        complexity: Sentence complexity level
        
    Returns:
        Tuple of (challenge_id, sentence)
    """
    generator = SentenceGenerator(complexity)
    return generator.generate()


if __name__ == "__main__":
    # Demo usage
    print("Simple sentences:")
    gen = SentenceGenerator("simple")
    for _ in range(3):
        cid, sentence = gen.generate()
        print(f"  {sentence}")
    
    print("\nMedium sentences:")
    gen = SentenceGenerator("medium")
    for _ in range(3):
        cid, sentence = gen.generate()
        print(f"  {sentence}")
    
    print("\nComplex sentences:")
    gen = SentenceGenerator("complex")
    for _ in range(3):
        cid, sentence = gen.generate()
        print(f"  {sentence}")

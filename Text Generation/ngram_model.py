import numpy as np
import pandas as pd
import nltk
import random
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import re

# Download the required NLTK resources
nltk.download('punkt')

class NGramModel:
    def __init__(self, n=2):
        """Initialize an N-gram model with the specified n value.
        
        Args:
            n (int): The length of each n-gram. Default is 2 (bigram).
        """
        self.n = n
        self.model = defaultdict(Counter)
        self.vocab = set()
        
    def train(self, text):
        """Train the model on the given text.
        
        Args:
            text (str): The text corpus to train on.
        """
        # Tokenize and preprocess the text
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Clean and tokenize the sentence
            tokens = self._preprocess_text(sentence)
            
            # Update vocabulary
            self.vocab.update(tokens)
            
            # Generate n-grams and build the model
            if len(tokens) >= self.n:
                for i in range(len(tokens) - self.n + 1):
                    history = tuple(tokens[i:i+self.n-1])
                    next_word = tokens[i+self.n-1]
                    self.model[history][next_word] += 1
    
    def _preprocess_text(self, text):
        """Clean and tokenize the text.
        
        Args:
            text (str): The text to preprocess.
        
        Returns:
            list: A list of tokens.
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        return tokens
    
    def generate_text(self, seed_words, num_words=50, selection_method='weighted'):
        """Generate text using the trained n-gram model.
        
        Args:
            seed_words (list): A list of words to start the generation.
            num_words (int): The number of words to generate. Default is 50.
            selection_method (str): Method to select the next word. Options are 'weighted' or 'max'.
                                   Default is 'weighted'.
        
        Returns:
            str: The generated text.
        """
        if len(seed_words) < self.n - 1:
            raise ValueError(f"Seed words must contain at least {self.n-1} words for n={self.n}")
        
        # Initialize with seed words
        generated = seed_words.copy()
        
        # Generate the specified number of words
        for _ in range(num_words):
            history = tuple(generated[-(self.n-1):])
            
            # If this history isn't in our model, we can't generate the next word
            if history not in self.model or not self.model[history]:
                # Apply backoff to a shorter history or choose a random word
                if self.n > 2 and len(history) > 1:
                    shorter_history = history[1:]
                    if shorter_history in self.model and self.model[shorter_history]:
                        candidates = self.model[shorter_history]
                    else:
                        # Fallback to random choice from vocabulary
                        candidates = {word: 1 for word in self.vocab}
                else:
                    # Fallback to random choice from vocabulary
                    candidates = {word: 1 for word in self.vocab}
            else:
                candidates = self.model[history]
            
            # Select the next word based on the specified method
            if selection_method == 'max':
                # Choose the most likely next word
                next_word = max(candidates.items(), key=lambda x: x[1])[0]
            else:  # weighted random selection
                # Convert counts to probabilities
                total_count = sum(candidates.values())
                candidates_prob = {word: count/total_count for word, count in candidates.items()}
                
                # Weighted random choice
                words = list(candidates_prob.keys())
                probabilities = list(candidates_prob.values())
                next_word = np.random.choice(words, p=probabilities)
            
            generated.append(next_word)
        
        return ' '.join(generated)

    def evaluate_perplexity(self, test_text):
        """Calculate the perplexity of the model on test data.
        
        Args:
            test_text (str): The text to evaluate on.
            
        Returns:
            float: The perplexity score (lower is better).
        """
        sentences = sent_tokenize(test_text)
        log_probs = []
        token_count = 0
        
        for sentence in sentences:
            tokens = self._preprocess_text(sentence)
            token_count += len(tokens)
            
            for i in range(len(tokens) - self.n + 1):
                history = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                
                if history in self.model and next_word in self.model[history]:
                    # Calculate probability of this n-gram
                    count = self.model[history][next_word]
                    total = sum(self.model[history].values())
                    probability = count / total
                    log_probs.append(np.log2(probability))
                else:
                    # Smoothing: assign a small probability for unseen n-grams
                    log_probs.append(np.log2(1 / (len(self.vocab) + 1)))
        
        # Calculate perplexity
        if not log_probs:
            return float('inf')
        
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity

def load_ag_news():
    """Load the AG News dataset from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("ag_news")
        # Combine title and text to form the corpus
        train_texts = [f"{item['title']} {item['text']}" for item in dataset["train"]]
        return " ".join(train_texts)
    except:
        # Fallback to a smaller sample if loading fails
        print("Failed to load AG News dataset. Using a sample text instead.")
        return """The technology industry continues to evolve rapidly with innovations in AI and robotics.
                 Climate change remains a significant global challenge as temperatures rise worldwide.
                 Economic indicators show mixed signals as inflation concerns persist in major markets.
                 Sports teams prepare for upcoming championships with intensive training sessions.
                 Healthcare advances include new treatments for chronic diseases and improved diagnostics.""" * 20

if __name__ == "__main__":
    # Load corpus
    corpus = load_ag_news()
    print(f"Corpus loaded with {len(corpus)} characters")
    
    # Split into train and test
    split_point = int(len(corpus) * 0.9)
    train_corpus = corpus[:split_point]
    test_corpus = corpus[split_point:]
    
    # Train models with different n values
    n_values = [1, 2, 3, 4]
    models = {}
    
    for n in n_values:
        print(f"\nTraining {n}-gram model...")
        model = NGramModel(n=n)
        model.train(train_corpus)
        models[n] = model
        
        # Evaluate model
        perplexity = model.evaluate_perplexity(test_corpus)
        print(f"Perplexity of {n}-gram model: {perplexity:.2f}")
    
    # Generate text with different seed sentences
    seed_sentences = [
        "the technology industry is",
        "global warming causes",
        "investors are concerned about"
    ]
    
    for n in [2, 3, 4]:  # Skip unigram as it doesn't use seed words in the same way
        print(f"\n===== Text generated with {n}-gram model =====")
        model = models[n]
        
        for seed in seed_sentences:
            seed_words = word_tokenize(seed.lower())
            
            # Ensure we have enough seed words for this n-gram model
            if len(seed_words) < n - 1:
                seed_words = seed_words + ["the"] * (n - 1 - len(seed_words))
            
            # Generate with weighted random selection
            generated_text = model.generate_text(
                seed_words=seed_words,
                num_words=30,
                selection_method='weighted'
            )
            print(f"\nSeed: '{seed}'")
            print(f"Generated (weighted): {generated_text}")
            
            # Generate with maximum likelihood
            generated_text = model.generate_text(
                seed_words=seed_words,
                num_words=30,
                selection_method='max'
            )
            print(f"Generated (max): {generated_text}") 
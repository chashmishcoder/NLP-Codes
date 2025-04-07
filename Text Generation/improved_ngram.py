import numpy as np
import pandas as pd
import nltk
import random
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import re
import math
from ngram_model import load_ag_news

# Download the required NLTK resources
nltk.download('punkt')

class ImprovedNGramModel:
    def __init__(self, n=2, smoothing='kneser_ney', discount=0.75):
        """Initialize an improved N-gram model with the specified parameters.
        
        Args:
            n (int): The length of each n-gram. Default is 2 (bigram).
            smoothing (str): Smoothing method to use: 'kneser_ney', 'interpolation', or 'backoff'.
            discount (float): Discount parameter for Kneser-Ney smoothing.
        """
        self.n = n
        self.smoothing = smoothing
        self.discount = discount
        self.model = defaultdict(Counter)
        self.vocab = set()
        self.context_counts = Counter()  # Count of contexts for Kneser-Ney
        self.continuation_counts = Counter()  # For Kneser-Ney smoothing
        self.lower_order_model = None  # For backoff smoothing
        self.total_tokens = 0
        
    def train(self, text):
        """Train the model on the given text.
        
        Args:
            text (str): The text corpus to train on.
        """
        # Tokenize and preprocess the text
        sentences = sent_tokenize(text)
        all_tokens = []
        
        for sentence in sentences:
            # Clean and tokenize the sentence
            tokens = self._preprocess_text(sentence)
            all_tokens.extend(tokens)
            
            # Update vocabulary
            self.vocab.update(tokens)
            
            # Generate n-grams and build the model
            if len(tokens) >= self.n:
                for i in range(len(tokens) - self.n + 1):
                    history = tuple(tokens[i:i+self.n-1])
                    next_word = tokens[i+self.n-1]
                    self.model[history][next_word] += 1
                    
                    # For Kneser-Ney smoothing
                    self.context_counts[history] += 1
                    self.continuation_counts[next_word] += 1
        
        self.total_tokens = len(all_tokens)
        
        # Train lower-order model for backoff and interpolation
        if self.n > 1 and (self.smoothing == 'backoff' or self.smoothing == 'interpolation'):
            self.lower_order_model = ImprovedNGramModel(n=self.n-1, smoothing=self.smoothing)
            self.lower_order_model.train(text)
    
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
    
    def _kneser_ney_probability(self, history, word):
        """Calculate Kneser-Ney smoothed probability for a word given its history.
        
        Args:
            history (tuple): The context (n-1 preceding words).
            word (str): The word to calculate probability for.
            
        Returns:
            float: The smoothed probability.
        """
        # If history not seen, use uniform distribution over vocabulary
        if history not in self.model or not self.model[history]:
            return 1.0 / len(self.vocab) if self.vocab else 0
        
        count = self.model[history][word]
        context_total = sum(self.model[history].values())
        
        # Calculate continuation count (for lower-order model)
        continuation_count = len(self.model[history])
        
        # Calculate Kneser-Ney probability
        if count > 0:
            # Apply discount
            prob = max(0, count - self.discount) / context_total
        else:
            prob = 0
            
        # Add the backoff/continuation probability
        lambda_factor = self.discount * continuation_count / context_total
        continuation_prob = self.continuation_counts[word] / len(self.continuation_counts)
        
        return prob + lambda_factor * continuation_prob
    
    def _interpolation_probability(self, history, word):
        """Calculate interpolated probability for a word given its history.
        
        Args:
            history (tuple): The context (n-1 preceding words).
            word (str): The word to calculate probability for.
            
        Returns:
            float: The interpolated probability.
        """
        lambda1 = 0.7  # Weight for the higher order model
        lambda2 = 0.3  # Weight for the lower order model
        
        higher_prob = 0
        if history in self.model:
            count = self.model[history][word]
            total = sum(self.model[history].values())
            higher_prob = count / total if total > 0 else 0
            
        # Get probability from lower-order model
        lower_prob = 0
        if self.lower_order_model is not None and len(history) > 0:
            lower_history = history[1:] if len(history) > 1 else ()
            lower_prob = self.lower_order_model._get_probability(lower_history, word)
        else:
            # Unigram case
            lower_prob = self.vocab and (1.0 / len(self.vocab)) or 0
            
        # Combine probabilities
        return lambda1 * higher_prob + lambda2 * lower_prob
    
    def _backoff_probability(self, history, word):
        """Calculate backoff probability for a word given its history.
        
        Args:
            history (tuple): The context (n-1 preceding words).
            word (str): The word to calculate probability for.
            
        Returns:
            float: The backoff probability.
        """
        # If we have this n-gram, use its probability
        if history in self.model and word in self.model[history]:
            count = self.model[history][word]
            total = sum(self.model[history].values())
            return count / total
        
        # Otherwise, back off to the lower-order model
        alpha = 0.4  # Backoff weight
        if self.lower_order_model is not None and len(history) > 0:
            lower_history = history[1:] if len(history) > 1 else ()
            return alpha * self.lower_order_model._get_probability(lower_history, word)
        else:
            # Unigram case
            return self.vocab and (alpha / len(self.vocab)) or 0
    
    def _get_probability(self, history, word):
        """Get probability based on selected smoothing method.
        
        Args:
            history (tuple): The context (n-1 preceding words).
            word (str): The word to calculate probability for.
            
        Returns:
            float: The probability.
        """
        if self.smoothing == 'kneser_ney':
            return self._kneser_ney_probability(history, word)
        elif self.smoothing == 'interpolation':
            return self._interpolation_probability(history, word)
        elif self.smoothing == 'backoff':
            return self._backoff_probability(history, word)
        else:
            # Default case: simple maximum likelihood
            if history in self.model:
                count = self.model[history][word]
                total = sum(self.model[history].values())
                return count / total if total > 0 else 0
            return 0
    
    def generate_text(self, seed_words, num_words=50, temperature=1.0):
        """Generate text using the improved n-gram model with temperature sampling.
        
        Args:
            seed_words (list): A list of words to start the generation.
            num_words (int): The number of words to generate. Default is 50.
            temperature (float): Controls randomness. Higher values (e.g., 1.0) make
                                the model more random, while lower values (e.g., 0.1)
                                make it more deterministic.
        
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
            
            # Get probabilities for all possible next words
            candidates = {}
            
            # If this history is in our model
            if history in self.model:
                for word in self.model[history]:
                    candidates[word] = self._get_probability(history, word)
            
            # If candidates is empty (unseen history), sample from vocabulary
            if not candidates:
                if self.n > 1 and self.lower_order_model is not None:
                    # Try backing off to a shorter history
                    shorter_history = history[1:] if len(history) > 1 else ()
                    for word in self.vocab:
                        candidates[word] = self.lower_order_model._get_probability(shorter_history, word)
                else:
                    # Fallback to uniform distribution over vocabulary
                    candidates = {word: 1.0/len(self.vocab) for word in self.vocab}
            
            # Apply temperature to control randomness
            if temperature != 1.0:
                # Adjust probabilities with temperature
                candidates = {w: math.pow(p, 1.0/temperature) for w, p in candidates.items()}
                
                # Normalize probabilities
                total = sum(candidates.values())
                if total > 0:
                    candidates = {w: p/total for w, p in candidates.items()}
            
            # Convert to lists for weighted choice
            words = list(candidates.keys())
            probabilities = list(candidates.values())
            
            # Normalize if needed
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p/prob_sum for p in probabilities]
            else:
                # If all probabilities are zero, use uniform distribution
                probabilities = [1.0/len(words) for _ in words] if words else []
            
            # Select next word
            if words and probabilities:
                next_word = np.random.choice(words, p=probabilities)
                generated.append(next_word)
            else:
                # If we don't have any valid options, pick a random word from vocabulary
                if self.vocab:
                    generated.append(random.choice(list(self.vocab)))
                else:
                    break
        
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
        
        for sentence in sentences:
            tokens = self._preprocess_text(sentence)
            
            for i in range(len(tokens) - self.n + 1):
                history = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                
                # Get probability using smoothing
                prob = self._get_probability(history, next_word)
                
                # Add a small epsilon to avoid log(0)
                epsilon = 1e-10
                log_probs.append(np.log2(prob + epsilon))
        
        # Calculate perplexity
        if not log_probs:
            return float('inf')
        
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity

def compare_smoothing_methods():
    """Compare the performance of different smoothing methods."""
    # Load corpus
    corpus = load_ag_news()
    print(f"Corpus loaded with {len(corpus)} characters")
    
    # Split into train and test
    split_point = int(len(corpus) * 0.9)
    train_corpus = corpus[:split_point]
    test_corpus = corpus[split_point:]
    
    # Define smoothing methods to compare
    smoothing_methods = ['kneser_ney', 'interpolation', 'backoff']
    
    # Initialize results storage
    results = []
    
    # Test with n=2 and n=3
    for n in [2, 3]:
        print(f"\nTesting with n={n}:")
        
        for method in smoothing_methods:
            print(f"  Training with {method} smoothing...")
            model = ImprovedNGramModel(n=n, smoothing=method)
            model.train(train_corpus)
            
            # Evaluate perplexity
            perplexity = model.evaluate_perplexity(test_corpus)
            print(f"  Perplexity with {method} smoothing: {perplexity:.2f}")
            
            # Generate sample text
            seed = "the technology industry is".lower().split()
            if len(seed) < n - 1:
                seed = seed + ["the"] * (n - 1 - len(seed))
                
            # Generate with different temperatures
            for temp in [0.5, 1.0, 1.5]:
                generated = model.generate_text(seed, num_words=30, temperature=temp)
                print(f"\n  Generated with {method}, temperature={temp}:")
                print(f"  {' '.join(seed)} {generated[len(' '.join(seed))+1:]}")
            
            # Store results
            results.append({
                'n': n,
                'smoothing': method,
                'perplexity': perplexity
            })
    
    return results

if __name__ == "__main__":
    results = compare_smoothing_methods() 
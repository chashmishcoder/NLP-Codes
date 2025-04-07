import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
from ngram_model import NGramModel, load_ag_news

# Ensure NLTK resources are downloaded
nltk.download('punkt')

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts (overlap of words)."""
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

def analyze_repetition(text):
    """Analyze repetition in generated text."""
    tokens = word_tokenize(text.lower())
    # Count bigrams
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_counts = Counter(bigrams)
    
    # Count trigrams
    trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    trigram_counts = Counter(trigrams)
    
    # Calculate repetition scores
    if len(tokens) <= 1:
        return 0, 0
    
    bigram_repetition = sum(count for count in bigram_counts.values() if count > 1) / len(bigrams) if bigrams else 0
    trigram_repetition = sum(count for count in trigram_counts.values() if count > 1) / len(trigrams) if trigrams else 0
    
    return bigram_repetition, trigram_repetition
def compare_models():
    """Train and compare N-gram models with different values of N."""
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
    perplexities = []
    
    for n in n_values:
        print(f"\nTraining {n}-gram model...")
        model = NGramModel(n=n)
        model.train(train_corpus)
        models[n] = model
        
        # Evaluate model
        perplexity = model.evaluate_perplexity(test_corpus)
        perplexities.append(perplexity)
        print(f"Perplexity of {n}-gram model: {perplexity:.2f}")
    
    # Plot perplexity comparison
    plt.figure(figsize=(10, 6))
    plt.bar(n_values, perplexities, color='skyblue')
    plt.xlabel('N-gram Size (N)')
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Perplexity of N-gram Models with Different N Values')
    plt.xticks(n_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('perplexity_comparison.png')
    print("Saved perplexity comparison plot to 'perplexity_comparison.png'")
    
    return models
def compare_generation_methods(models):
    """Compare different text generation methods across models."""
    seed_sentences = [
        "the technology industry is",
        "global warming causes",
        "investors are concerned about"
    ]
    
    # Store results for analysis
    jaccard_scores = {n: [] for n in [2, 3, 4]}
    repetition_scores = {
        'weighted': {n: {'bigram': [], 'trigram': []} for n in [2, 3, 4]},
        'max': {n: {'bigram': [], 'trigram': []} for n in [2, 3, 4]}
    }
    
    for n in [2, 3, 4]:  # Skip unigram as it doesn't use seed words the same way
        model = models[n]
        
        print(f"\n===== Text generated with {n}-gram model =====")
        
        for seed in seed_sentences:
            seed_words = word_tokenize(seed.lower())
            
            # Ensure we have enough seed words for this n-gram model
            if len(seed_words) < n - 1:
                seed_words = seed_words + ["the"] * (n - 1 - len(seed_words))
            
            # Generate with weighted random selection
            weighted_text = model.generate_text(
                seed_words=seed_words,
                num_words=50,
                selection_method='weighted'
            )
            
            # Generate with maximum likelihood
            max_text = model.generate_text(
                seed_words=seed_words,
                num_words=50,
                selection_method='max'
            )
            
            print(f"\nSeed: '{seed}'")
            print(f"Generated (weighted): {weighted_text}")
            print(f"Generated (max): {max_text}")
            
            # Calculate Jaccard similarity between generation methods
            similarity = jaccard_similarity(weighted_text, max_text)
            jaccard_scores[n].append(similarity)
            
            # Analyze repetition
            w_bigram_rep, w_trigram_rep = analyze_repetition(weighted_text)
            m_bigram_rep, m_trigram_rep = analyze_repetition(max_text)
            
            repetition_scores['weighted'][n]['bigram'].append(w_bigram_rep)
            repetition_scores['weighted'][n]['trigram'].append(w_trigram_rep)
            repetition_scores['max'][n]['bigram'].append(m_bigram_rep)
            repetition_scores['max'][n]['trigram'].append(m_trigram_rep)
    
    # Plot Jaccard similarity
    plt.figure(figsize=(10, 6))
    for n in [2, 3, 4]:
        plt.plot(range(len(seed_sentences)), jaccard_scores[n], marker='o', label=f'N={n}')
    
    plt.xlabel('Seed Sentence Index')
    plt.ylabel('Jaccard Similarity')
    plt.title('Similarity Between Weighted and Max Selection Methods')
    plt.xticks(range(len(seed_sentences)), [s[:15] + '...' for s in seed_sentences], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('jaccard_similarity.png')
    print("Saved Jaccard similarity plot to 'jaccard_similarity.png'")
    
    # Plot repetition scores
    plt.figure(figsize=(12, 8))
    
    n_values = [2, 3, 4]
    x = np.arange(len(n_values))
    width = 0.2
    
    # Calculate averages
    w_bigram_avgs = [np.mean(repetition_scores['weighted'][n]['bigram']) for n in n_values]
    w_trigram_avgs = [np.mean(repetition_scores['weighted'][n]['trigram']) for n in n_values]
    m_bigram_avgs = [np.mean(repetition_scores['max'][n]['bigram']) for n in n_values]
    m_trigram_avgs = [np.mean(repetition_scores['max'][n]['trigram']) for n in n_values]
    
    plt.bar(x - width*1.5, w_bigram_avgs, width, label='Weighted Bigram', color='skyblue')
    plt.bar(x - width/2, w_trigram_avgs, width, label='Weighted Trigram', color='royalblue')
    plt.bar(x + width/2, m_bigram_avgs, width, label='Max Bigram', color='lightcoral')
    plt.bar(x + width*1.5, m_trigram_avgs, width, label='Max Trigram', color='firebrick')
    
    plt.xlabel('N-gram Size (N)')
    plt.ylabel('Repetition Score (higher means more repetition)')
    plt.title('Repetition in Generated Text by Model and Selection Method')
    plt.xticks(x, n_values)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('repetition_analysis.png')
    print("Saved repetition analysis plot to 'repetition_analysis.png'")

if __name__ == "__main__":
    models = compare_models()
    compare_generation_methods(models) 

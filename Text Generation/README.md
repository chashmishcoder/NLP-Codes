# N-Gram Text Generation Model

This project implements an N-gram model for text generation using the AG News dataset. The model supports different N values and text generation methods.

## Features

- Implementation of N-gram models with variable N (1, 2, 3, 4)
- Text generation using different seed sentences
- Two selection methods for next word prediction:
  - Weighted random selection (based on probabilities)
  - Maximum likelihood (selecting the most probable next word)
- Advanced smoothing techniques:
  - Kneser-Ney smoothing
  - Linear interpolation
  - Backoff modeling
- Temperature-based sampling for controlling randomness
- Perplexity evaluation for model performance
- Visual analysis of model performance and generated text characteristics

## Requirements

To run this project, you need Python 3.6+ and the following packages:

```
pip install -r requirements.txt
```

## Usage

### Basic N-gram Model

Run the main script to train models with different N values and generate text:

```
python ngram_model.py
```

The script will:
1. Download and load the AG News dataset (or use a fallback dataset if unavailable)
2. Train N-gram models with N = 1, 2, 3, and 4
3. Evaluate each model using perplexity
4. Generate text using different seed sentences and selection methods

### Improved N-gram Model

For an enhanced version with advanced smoothing techniques and temperature control:

```
python improved_ngram.py
```

This implementation includes:
- Kneser-Ney smoothing - sophisticated discounting technique
- Linear interpolation - combining probabilities from different order models
- Backoff modeling - falling back to lower-order models for unseen n-grams
- Temperature sampling - controlling randomness in text generation

### Analysis and Visualization

For a more detailed analysis with visualizations, run:

```
python ngram_analysis.py
```

This script performs:
1. Comparison of perplexity across different N values
2. Analysis of similarity between different text generation methods
3. Measurement of repetition in generated text
4. Visualization with plots saved as PNG files:
   - `perplexity_comparison.png`: Bar chart of perplexity scores
   - `jaccard_similarity.png`: Plot of text similarity between generation methods
   - `repetition_analysis.png`: Analysis of repetition patterns in generated text

## Customization

You can modify the following in the scripts:

- Change the N values by editing the `n_values` list
- Add or modify seed sentences in the `seed_sentences` list
- Adjust the number of generated words by changing the `num_words` parameter
- Use a different dataset by modifying the `load_ag_news()` function
- In the improved model:
  - Change smoothing methods (`kneser_ney`, `interpolation`, `backoff`)
  - Adjust the temperature parameter to control randomness

## Results and Evaluation

The results will show:
- Perplexity scores for each N-gram model (lower is better)
- Generated text for each seed sentence using different N values
- Comparison between weighted random selection and maximum likelihood methods
- Visual analysis of text quality metrics
- Effect of different smoothing techniques on text quality and perplexity

### Key Findings
- Higher N values typically produce more coherent text but are more prone to data sparsity
- Maximum likelihood selection often leads to more repetitive patterns
- Weighted random selection provides more diverse outputs
- Smoothing techniques help handle unseen word sequences
- Temperature controls the trade-off between diversity and coherence 

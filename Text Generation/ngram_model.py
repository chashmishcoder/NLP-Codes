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
    """Load the AG News dataset or create a larger local dataset."""
    try:
        print("Attempting to load AG News dataset...")
        from datasets import load_dataset
        dataset = load_dataset("ag_news")
        print(f"Successfully loaded AG News dataset with {len(dataset['train'])} training examples")
        
        # Print dataset structure for debugging
        print("Dataset structure:")
        print("Keys in dataset:", dataset.keys())
        print("Columns in train set:", dataset['train'].column_names)
        print("First example:", dataset['train'][0])
        
        # Correctly extract text based on the actual structure
        if 'text' in dataset['train'].column_names:
            # If dataset has both title and text fields
            if 'title' in dataset['train'].column_names:
                train_texts = [f"{item['title']} {item['text']}" for item in dataset['train']]
            else:
                train_texts = [item['text'] for item in dataset['train']]
        else:
            # Fall back to using the raw data as is
            train_texts = [str(item) for item in dataset['train']]
            
        corpus = " ".join(train_texts)
        print(f"Created corpus with {len(corpus)} characters")
        return corpus
    except Exception as e:
        print(f"Failed to load AG News dataset. Error: {e}")
        print("Creating a larger local dataset instead.")
        
        # Create a larger local dataset for better n-gram training
        local_corpus = """
        TECHNOLOGY NEWS:
        The technology industry continues to evolve rapidly with innovations in AI and robotics.
        Apple unveils new iPhone with enhanced AI capabilities and improved battery life.
        Microsoft announces significant updates to Windows operating system with new features.
        Google's latest algorithm update aims to improve search results for users worldwide.
        Amazon expands cloud computing services with new data centers in multiple regions.
        Tesla introduces advanced self-driving features for its electric vehicle lineup.
        SpaceX successfully launches rocket carrying satellites for global internet coverage.
        Cybersecurity experts warn of increasing ransomware attacks targeting businesses.
        New semiconductor technology promises faster and more energy-efficient computing.
        Social media platforms implement new features to combat misinformation and fake news.
        
        WORLD NEWS:
        Climate change remains a significant global challenge as temperatures rise worldwide.
        International summit on climate change concludes with new emissions reduction targets.
        Diplomatic tensions rise between major powers over disputed territorial claims.
        United Nations calls for increased humanitarian aid in conflict-affected regions.
        Peace negotiations continue in attempt to resolve long-standing regional conflicts.
        Global health organization reports decrease in communicable disease rates worldwide.
        International trade agreements face scrutiny amid changing economic relationships.
        Cultural exchange programs promote understanding between diverse nations and peoples.
        Environmental conservation efforts receive boost from multinational cooperation.
        World leaders discuss strategies to address migration challenges at annual forum.
        
        BUSINESS NEWS:
        Economic indicators show mixed signals as inflation concerns persist in major markets.
        Stock markets react to central bank decisions on interest rates and monetary policy.
        Investors are concerned about potential economic slowdown affecting global markets.
        Major merger between industry leaders reshapes competitive landscape in key sector.
        Startup companies attract record venture capital funding in emerging technology fields.
        Consumer spending trends indicate shifting preferences in post-pandemic economy.
        Supply chain disruptions continue to affect manufacturing and retail industries.
        Corporate sustainability initiatives gain prominence among leading businesses.
        Financial regulators propose new frameworks for cryptocurrency oversight.
        Labor market shows resilience despite economic uncertainties in various sectors.
        
        SPORTS NEWS:
        Sports teams prepare for upcoming championships with intensive training sessions.
        Olympic athletes break records in multiple events during international competition.
        Football club announces signing of star player in record-breaking transfer deal.
        Basketball tournament concludes with dramatic overtime victory in final game.
        Tennis champion defends title in grueling five-set match against top challenger.
        Golf tournament attracts worldwide audience as players compete for prestigious trophy.
        Racing team unveils new vehicle design with advanced aerodynamic features.
        Cricket match ends in historic result after exceptional individual performance.
        Swimming competition showcases emerging talent alongside established champions.
        Team sports emphasize importance of mental health support for professional athletes.
        
        HEALTH NEWS:
        Healthcare advances include new treatments for chronic diseases and improved diagnostics.
        Medical researchers announce breakthrough in understanding rare genetic disorders.
        Vaccine development accelerates for prevalent infectious diseases affecting populations.
        Public health campaigns focus on preventative measures for community wellbeing.
        Telemedicine services expand to provide greater access to healthcare specialists.
        Mental health awareness initiatives receive support from healthcare institutions.
        Nutritional studies reveal connections between diet and long-term health outcomes.
        Fitness trends emphasize personalized approaches to physical wellness and activity.
        Medical technology innovations improve surgical procedures and patient recovery.
        Healthcare systems implement reforms to address accessibility and affordability concerns.
        
        SCIENCE NEWS:
        Astronomical observations reveal new insights about distant planetary systems.
        Genetic research advances understanding of evolutionary relationships between species.
        Physics experiment confirms theoretical predictions about fundamental particles.
        Marine biologists document previously unknown species in deep ocean environments.
        Geological studies provide information about Earth's historical climate patterns.
        Artificial intelligence assists researchers in analyzing complex scientific data.
        Conservation efforts focus on protecting biodiversity in threatened ecosystems.
        Renewable energy research develops more efficient solar and wind power technologies.
        Neuroscience studies enhance knowledge of brain function and cognitive processes.
        Scientific collaboration across disciplines addresses complex global challenges.
        
        EDUCATION NEWS:
        Educational institutions implement innovative teaching methods using digital tools.
        Research universities announce expanded scholarship programs for diverse students.
        Online learning platforms provide accessible education to global participants.
        STEM education initiatives encourage student interest in technical career paths.
        Educational policy reforms address challenges in learning assessment and outcomes.
        Teacher training programs focus on inclusive approaches to classroom instruction.
        International education exchanges promote cross-cultural understanding and cooperation.
        Literacy programs show positive results in improving reading skills for young learners.
        Educational technology enhances student engagement with interactive learning materials.
        Lifelong learning becomes priority as workforce adapts to changing skill requirements.
        """
        print(f"Created local corpus with {len(local_corpus)} characters")
        return local_corpus



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

# src/model/ngram_model.py
from collections import Counter, defaultdict
import os
import json
from dotenv import load_dotenv
class NGramModel:
    """
    Responsible for building, storing, and exposing n-gram probability tables 
    and backoff lookup across all orders from 1 up to NGRAM_ORDER.
    """
    def __init__(self, unk_threshold: int, ngram_order: int):
        self.unk_threshold = unk_threshold
        self.ngram_order = ngram_order
        self.vocab = set()
        
        # Raw counts: {order: Counter(tuple: int)}
        self.counts = {order: Counter() for order in range(1, ngram_order + 1)}
        
        # Probabilities: {order: {context_tuple: {target_word: probability}}}
        self.probabilities = {order: defaultdict(dict) for order in range(1, ngram_order + 1)}

    def build_vocab(self, token_file: str):
        """
        Reads the tokenized file, builds a vocabulary of words meeting UNK_THRESHOLD,
        and ensures <UNK> is included.
        
        :param token_file: Path to the file containing tokenized sentences.
        """
        word_counts = Counter()
        
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token file not found: {token_file}")

        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                word_counts.update(tokens)
            
        self.vocab = {word for word, count in word_counts.items() 
                      if count >= self.unk_threshold}
        self.vocab.add("<UNK>")
        print(f"Vocab built. Size: {len(self.vocab)}")

    def build_counts_and_probabilities(self, token_file: str):
        """
        Counts all n-grams from orders 1 to NGRAM_ORDER and computes 
        Maximum Likelihood Estimation (MLE) probabilities.
        
        :param token_file: Path to the file containing tokenized sentences.
        """
        # 1. First Pass: Build all counts
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Apply UNK substitution on the fly
                sentence = [w if w in self.vocab else "<UNK>" for w in line.strip().split()]
                sent_len = len(sentence)
                
                max_possible_order = min(sent_len, self.ngram_order)
                for order in range(1, max_possible_order + 1):
                    for i in range(sent_len - order + 1):
                        ngram = tuple(sentence[i : i + order])
                        self.counts[order][ngram] += 1

        # 2. Second Pass: Compute MLE Probabilities
        # Formula: P(w | context) = count(context + w) / count(context)
        for order in range(1, self.ngram_order + 1):
            for ngram, count in self.counts[order].items():
                if order == 1:
                    # Unigram probability: count(word) / total_words
                    total_unigrams = sum(self.counts[1].values())
                    self.probabilities[1][()][ngram[0]] = count / total_unigrams
                else:
                    # Higher order: context is the first (n-1) words
                    context = ngram[:-1]
                    target = ngram[-1]
                    context_count = self.counts[order - 1][context]
                    
                    if context_count > 0:
                        self.probabilities[order][context][target] = count / context_count

        print(f"Probabilities computed for all orders up to {self.ngram_order}.")
    
    def lookup(self, context: tuple) -> dict:
        """
        Performs a backoff lookup. It tries the longest possible context first.
        If no matches are found, it 'backs off' by dropping the first word 
        of the context and trying again, down to 1-grams.
        
        :param context: A tuple of words (the prefix).
        :return: A dictionary of {next_word: probability} or an empty dict.
        """
        # We start looking at the highest possible order allowed by our model
        # If the context is ("a", "b", "c") and NGRAM_ORDER is 4, we start at order 4.
        current_order = min(len(context) + 1, self.ngram_order)
        
        while current_order >= 1:
            # 1-grams have an empty tuple as context ()
            search_context = context[-(current_order - 1):] if current_order > 1 else ()
            
            # Check if this context exists in our probability table
            if search_context in self.probabilities[current_order]:
                return self.probabilities[current_order][search_context]
            
            # Backoff: move to a smaller N-gram order
            current_order -= 1

        return {}

    def save_model(self, model_path: str):
        """
        Serializes the probability tables into a JSON file.
        Converts tuple keys to space-separated strings for JSON compatibility.
        """
        export_data = {}
        
        for order in range(1, self.ngram_order + 1):
            key_name = f"{order}gram"
            # JSON keys must be strings, so we join our tuples with spaces
            # Unigrams (order 1) have an empty context, so we handle them specially
            if order == 1:
                export_data[key_name] = self.probabilities[1][()]
            else:
                export_data[key_name] = {
                    " ".join(ctx): targets 
                    for ctx, targets in self.probabilities[order].items()
                }

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Model saved successfully to {model_path}")

    def save_vocab(self, vocab_path: str):
        """
        Serializes the vocabulary set into a JSON list.
        
        :param vocab_path: Path where the vocab.json should be saved.
        """
        # Convert set to a sorted list for consistent file output
        vocab_list = sorted(list(self.vocab))
        
        try:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_list, f, indent=2)
            print(f"Vocabulary saved successfully to {vocab_path}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")
    def lookkup(self, context: tuple) -> dict:
        """
        Performs a backoff lookup compatible with JSON string keys.
        
        :param context: A tuple of words (the prefix).
        :return: A dictionary of {next_word: probability} or an empty dict.
        """
        # Start at the highest possible order (N)
        current_order = min(len(context) + 1, self.ngram_order)
        
        while current_order >= 1:
            # 1. Create the key name that matches model.json (e.g., "4gram")
            order_key = f"{current_order}gram"
            
            # 2. Get the specific number of words needed for this order
            # For order 4, we need 3 words of context
            sub_context_tuple = context[-(current_order - 1):] if current_order > 1 else ()
            
            # 3. Convert the tuple to a space-separated string to match JSON keys
            search_context_str = " ".join(sub_context_tuple)
            
            # 4. Check if the order and the context exist in our probabilities
            if order_key in self.probabilities:
                if search_context_str in self.probabilities[order_key]:
                    return self.probabilities[order_key][search_context_str]
            
            # Backoff: move to a smaller N-gram order (e.g., 4gram -> 3gram)
            current_order -= 1

        return {}
        """
        Modified lookup to handle both training (tuples) and 
        loaded JSON models (strings).
        """
        # Ensure context is a list of words to work with
        if isinstance(context, str):
            context_words = context.split()
        else:
            context_words = list(context)

        current_order = min(len(context_words) + 1, self.ngram_order)

        while current_order >= 1:
            # 1. Get the slice of words for this order
            slice_of_words = context_words[-(current_order - 1):] if current_order > 1 else []
            
            # 2. Create the string key "word1 word2"
            search_context = " ".join(slice_of_words)
            
            # 3. Check both int and str keys (JSON loads keys as strings like "4")
            order_key = current_order
            str_order_key = f"{order_key}gram" # Matches your save_model "4gram" format
            
            # Look in the loaded probabilities
            if str_order_key in self.probabilities:
                if search_context in self.probabilities[str_order_key]:
                    return self.probabilities[str_order_key][search_context]
            
            current_order -= 1

        return {}


if __name__ == "__main__":
    # 1. Create a dummy token file for testing
    load_dotenv('config/.env')
    token_path = os.getenv("TRAIN_TOKENS")
    model_file = os.getenv("MODEL")
    vocab_file = os.getenv("VOCAB")

    # 2. Initialize
    model = NGramModel(unk_threshold=3, ngram_order=4)
    model.build_vocab(token_path)
    model.build_counts_and_probabilities(token_path)

    model.save_vocab("vocab_file")
    model.save_model("model_file")
    test_context = ("is", "always", "the")
    print(f"Testing context: {test_context}")
        
    results = model.lookup(test_context)
    print(f"result: {results}")
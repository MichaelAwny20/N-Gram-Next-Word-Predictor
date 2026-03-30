import os
from dotenv import load_dotenv

class Predictor:
    def __init__(self, model, normalizer):
        """
        :param model: An instance of NGramModel (loaded from JSON).
        :param normalizer: An instance of Normalizer.
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text: str) -> tuple:
        """Cleans input and extracts the last N-1 words as a TUPLE."""
        tokens = self.normalizer.normalize(text)
        
        # Access the ngram_order from the model (e.g., 4)
        context_size = self.model.ngram_order - 1
        
        # Return as tuple to allow proper word-based slicing later
        return tuple(tokens.split(" ")[-context_size:]) if tokens else ()

    def map_oov(self, context: tuple) -> tuple:
        """Replaces words not in the model's vocabulary list with <UNK>."""
        # Works perfectly even if self.model.vocab is a JSON list
        return tuple(w if w in self.model.vocab else "<UNK>" for w in context)

    def predict_next(self, text: str, k: int) -> list:
        """Main method: Normalize -> Map OOV -> Tuple Lookup -> Rank."""
        # 1. Get the last N-1 words as a tuple
        raw_context = self.normalize(text)
        
        # 2. Convert unknown words to <UNK>
        safe_context = self.map_oov(raw_context)
        
        # 3. Pass the TUPLE to the model's lookup
        predictions_dict = self.model.lookup(safe_context)
        
        
        if not predictions_dict:
            return []
            
        # 4. Sort by probability descending
        sorted_predictions = sorted(
            predictions_dict.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        return [word for word, prob in sorted_predictions[:k]]

# --- TEST BLOCK: Run this file directly to verify ---
if __name__ == "__main__":
    class MockNormalizer:
        def normalize(self, text):
            import re
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text.split()

    class MockModel:
        def __init__(self):
            self.ngram_order = 4
            # VOCAB is a LIST (just like your JSON)
            self.vocab = ["she", "is", "always", "the", "woman", "most", "first", "<UNK>"]
            
            # PROBABILITIES use string keys (just like your JSON)
            self.probabilities = {
                "4gram": {"is always the": {"woman": 0.33, "most": 0.33, "first": 0.33}},
                "3gram": {"always the": {"case": 1.0}},
                "2gram": {"the": {"end": 0.5}},
                "1gram": {"": {"the": 0.1}}
            }

        def lookup(self, context_tuple):
            """
            Handles the backoff logic by slicing the TUPLE 
            and converting to a STRING only for the dictionary key.
            """
            current_order = min(len(context_tuple) + 1, self.ngram_order)
            
            while current_order >= 1:
                # 1. Use the JSON-style string key for the order
                order_key = f"{current_order}gram"
                
                # 2. Slice the tuple to get the context for this order
                # (current_order - 1) words of context
                sub_context = context_tuple[-(current_order-1):] if current_order > 1 else ()
                
                # 3. Convert tuple to space-separated string to match JSON keys
                search_key = " ".join(sub_context)
                
                if order_key in self.probabilities and search_key in self.probabilities[order_key]:
                    return self.probabilities[order_key][search_key]
                
                current_order -= 1
            return {}

    # Initialize
    predictor = Predictor(MockModel(), MockNormalizer())
    
    # Test Case
    test_phrase = "always the"
    results = predictor.predict_next(test_phrase, k=3)
    
    print(f"Input: '{test_phrase}'")
    print(f"Predictions: {results}")
# main.py
import argparse
import os
from dotenv import load_dotenv

# Import your custom modules
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    # 1. Load configuration from .env
    load_dotenv("config/.env")
    
    # Get settings (with defaults if .env is missing)
    ngram_order = int(os.getenv("NGRAM_ORDER", 4))
    unk_threshold = int(os.getenv("UNK_THRESHOLD", 3))
    top_k = int(os.getenv("TOP_K", 3))
    raw_data = os.getenv("TRAIN_RAW_DIR")
    token_path = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")

    
    # 2. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Sherlock N-Gram Text Predictor")
    parser.add_argument("--step", choices=["dataprep", "model", "inference", "all"], 
                        required=True, help="Which part of the pipeline to run")
    args = parser.parse_args()

    # 3. Initialize the core components
    # (We do this once so they are available for any step)
    norm = Normalizer()
    model = NGramModel(unk_threshold=unk_threshold, ngram_order=ngram_order)
    
    # --- STEP: DATAPREP ---
    if args.step in ["dataprep", "all"]:
        print("--- Running Data Preparation ---")
        # Assuming you have a method in Normalizer to process a whole file
        # If not, you can simply read raw_data and write to token_path here
        print(f"Cleaning {raw_data}...")
        # (Logic to read raw, normalize, and save to tokens goes here)
        files_loaded = norm.load(raw_data)
        files_prepared = norm.process_all(files_loaded)
        
        norm.save(files_prepared,token_path)
        print(f"Tokens saved to {token_path}")

    # --- STEP: MODEL ---
    if args.step in ["model", "all"]:
        print("--- Building Model ---")
        model.build_vocab(token_path)
        model.build_counts_and_probabilities(token_path)
        model.save_vocab(vocab_path)
        model.save_model(model_path)

    # --- STEP: INFERENCE ---
    # --- STEP: INFERENCE ---
    if args.step in ["inference", "all"]:
        print("\n--- Starting Inference Loop ---")
        
        # CRITICAL: If we are ONLY running inference, we MUST load the files
        if args.step == "inference":
            print(f"Loading model from {model_path}...")
            with open(model_path, 'r', encoding='utf-8') as f:
                model.probabilities = json.load(f)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                model.vocab = set(json.load(f))
        
        predictor = Predictor(model, norm)
        # ... rest of your loop
        
        try:
            while True:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye.")
                    break
                
                if not user_input:
                    continue
                predictions = predictor.predict_next(user_input, k=top_k)
                print(f"Predictions: {predictions}")
                
                
        except KeyboardInterrupt:
            print("\nExiting cleanly... Goodbye.")

if __name__ == "__main__":
    main()
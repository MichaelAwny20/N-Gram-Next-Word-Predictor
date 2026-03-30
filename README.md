# Sherlock Holmes Next-Word Predictor

A modular N-Gram language model built as a Capstone project.

## Requirements
- `python version`: Python 3.10 or higher version.

## Project Structure
- `src/data_prep`: Text normalization and cleaning.
- `src/model`: N-Gram frequency mapping and training.
- `src/inference`: Prediction logic with backoff.
- `main.py`: CLI Entry point.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place Sherlock Holmes .txt files in `data/raw/train/`
3. Run the project: `python main.py`
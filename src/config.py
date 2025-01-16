import os

MODEL_NAME = os.getenv("MODEL_NAME", "google-bert/bert-base-uncased")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", MODEL_NAME)
DATASET_NAME = os.getenv("DATASET_NAME", "arad1367/Crypto_Fundamental_News")
NUM_LABELS = int(os.getenv("NUM_LABELS", 3))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results")

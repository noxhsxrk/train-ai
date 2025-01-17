import os

# MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
MODEL_NAME = os.getenv("MODEL_NAME", "PrunaAI/KBTG-Labs-THaLLE-0.1-7B-fa-bnb-4bit-smashed")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", MODEL_NAME)
DATASET_NAME = os.getenv("DATASET_NAME", "arad1367/Crypto_Fundamental_News")
NUM_LABELS = int(os.getenv("NUM_LABELS", 3))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results")

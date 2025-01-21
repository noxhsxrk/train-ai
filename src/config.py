import os

MODEL_NAME = os.getenv("MODEL_NAME", "openai-community/gpt2")
# MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN", "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")
DATASET_NAME = os.getenv("DATASET_NAME", "arad1367/Crypto_Fundamental_News")
NUM_LABELS = int(os.getenv("NUM_LABELS", 3))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results")

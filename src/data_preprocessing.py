from datasets import load_dataset
from transformers import AutoTokenizer
import src.config as config

def load_data():
    """
    Load the dataset from Hugging Face.
    """
    dataset = load_dataset(config.DATASET_NAME)
    return dataset

def preprocess_data(dataset, tokenizer):
    """
    Tokenization and preprocess dataset.
    """
    def preprocess_function(examples):
        # Map labels from string to integers
        labels = [
            0 if label == "neutral" else 1 if label == "positive" else 2
            for label in examples["label"]
        ]
        # Tokenize inputs
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        tokenized["labels"] = labels
        return tokenized

    # Apply tokenization and preprocessing
    dataset = dataset.map(preprocess_function, batched=True)

    # Remove unnecessary columns
    dataset = dataset.remove_columns(["text", "label"])

    # Set dataset format for PyTorch
    dataset.set_format("torch")
    return dataset

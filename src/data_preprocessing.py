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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def preprocess_function(examples):
        labels = [
            0 if label == "neutral" else 1 if label == "positive" else 2
            for label in examples["label"]
        ]
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.remove_columns(["text", "label"])
    dataset.set_format("torch")
    return dataset

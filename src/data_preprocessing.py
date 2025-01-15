from datasets import load_dataset
from transformers import AutoTokenizer
import src.config as config

def load_data():
    """
    Load the dataset from Hugging Face.
    """
    dataset = load_dataset(config.DATASET_NAME)
    return dataset

def preprocess_data(dataset, tokenizer_name):
    """
    Tokenization and preprocess dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        # Tokenize the text and map labels to numerical values
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized["labels"] = [
            0 if label == "neutral" else 1 if label == "positive" else 2
            for label in examples["label"]
        ]
        return tokenized

    # Apply tokenization and preprocessing
    dataset = dataset.map(preprocess_function, batched=True)

    # Remove unnecessary columns
    dataset = dataset.remove_columns(["text", "label"])

    # Set dataset format for PyTorch
    dataset.set_format("torch")
    return dataset
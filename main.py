from src.data_preprocessing import load_data, preprocess_data
from src.model import load_model
from src.train import train_model
import src.config as config
from transformers import AutoTokenizer

def main():
    dataset = load_data()
    
    train_val_split = dataset["train"].train_test_split(test_size=0.2)
    train_data = train_val_split["train"]
    validation_data = train_val_split["test"]

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_data = preprocess_data(train_data, tokenizer)
    validation_data = preprocess_data(validation_data, tokenizer)

    model = load_model(config.MODEL_NAME, config.NUM_LABELS)
    model.resize_token_embeddings(len(tokenizer))

    trainer = train_model(model, train_data, validation_data, tokenizer, config.OUTPUT_DIR)
    
    metrics = trainer.evaluate(validation_data)
    print(metrics)

if __name__ == "__main__":
    main()

from src.data_preprocessing import load_data, preprocess_data
from src.model import load_model
from src.train import train_model
import src.config as config

def main():
    dataset = load_data()
    
    train_val_split = dataset["train"].train_test_split(test_size=0.2)
    train_data = train_val_split["train"]
    validation_data = train_val_split["test"]

    train_data = preprocess_data(train_data, config.TOKENIZER_NAME)
    validation_data = preprocess_data(validation_data, config.TOKENIZER_NAME)

    model = load_model(config.MODEL_NAME, config.NUM_LABELS, True)

    train_model(model, train_data, validation_data, config.TOKENIZER_NAME, config.OUTPUT_DIR)

if __name__ == "__main__":
    main()

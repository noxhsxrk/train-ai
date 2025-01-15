from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

def load_model(model_name, num_labels, quantization=False):
    """
    Load model for sequence classification with optional quantization.
    """
    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            quantization_config=quantization_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    return model
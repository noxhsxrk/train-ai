from transformers import AutoModelForSequenceClassification
import torch
import src.config as config

def load_model(model_name, num_labels):
    """
    Load model for sequence classification with optional quantization.
    Automatically fallback to CPU or MPS if GPU is unavailable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        token=config.HF_TOKEN,
        num_labels=num_labels,
    )

    
    model.to(device)
    print(f"Model loaded on device: {device}")
    model.gradient_checkpointing_enable()
    return model

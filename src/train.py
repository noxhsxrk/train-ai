from transformers import TrainingArguments, Trainer

def train_model(model, train_data, validation_data, tokenizer, output_dir="./results"):
    """
    Train model.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
import src.config as config

def train_model(model, train_data, validation_data, tokenizer, output_dir=config.OUTPUT_DIR):
    """
    Train model with error handling for missing labels.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # load best model
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,           # save only 1 check point
        # fp16=True                     # use mixed precision (require GPU)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator,
    )

    trainer.train()
    return 

from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
import src.config as config

def train_model(model, train_data, validation_data, tokenizer, output_dir=config.OUTPUT_DIR):
    """
    Train model with error handling for missing labels.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,           # save only 1 check point
        load_best_model_at_end=True,  # load best model
        fp16=True                     # use mixed precision (GPU only)
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

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "jhenberthf/filipino-gossip-dataset"
MAX_LENGTH = 1024
OUTPUT_DIR = "./marites_model"

# Training arguments with better defaults
TRAIN_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "bf16": torch.cuda.is_available(),
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
}

class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            if torch.isinf(torch.tensor(logs["loss"])) or torch.isnan(torch.tensor(logs["loss"])):
                print("\n⚠️ Numerical instability detected! Stopping training.")
                control.should_training_stop = True
                
            if logs["loss"] > 1000:  # High loss threshold
                print("\n⚠️ High loss detected! Running diagnostics...")
                self._run_model_checks(kwargs['model'], kwargs['tokenizer'])

    def _run_model_checks(self, model, tokenizer):
        # Basic model output check
        test_text = "Bakit daw nag resign si Miss Lorna sa opisina?"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            print("✅ Model forward pass successful")
            print(f"Logits shape: {outputs.logits.shape}")
        except Exception as e:
            print(f"❌ Model forward pass failed: {str(e)}")

# Load and prepare dataset
def prepare_dataset():
    dataset = load_dataset(DATASET_NAME)
    
    # Handle dataset splits
    if isinstance(dataset, dict) and 'train' in dataset:
        return dataset['train'].train_test_split(test_size=0.1)
    return dataset.train_test_split(test_size=0.1)

# Initialize components
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Dataset processing
def format_text(examples):
    return {"text": [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False
        ) for p, r in zip(examples['prompt'], examples['response'])
    ]}

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # Dynamic padding handled by collator
        add_special_tokens=True
    )

# Main execution
if __name__ == "__main__":
    # Prepare and tokenize dataset
    dataset = prepare_dataset()
    dataset = dataset.map(format_text, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Model initialization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        use_cache=False
    )

    # Pre-training checks
    print("\nRunning initial validation:")
    sample_text = "Bakit daw nag resign si Miss Lorna sa opisina?"
    sample_input = tokenizer(sample_text, return_tensors="pt").to(model.device)
    sample_input["labels"] = sample_input["input_ids"]  # Add labels for loss calculation
    
    with torch.no_grad():
        outputs = model(**sample_input)
        initial_loss = outputs.loss.item() if outputs.loss else float('inf')
        print(f"Initial validation loss: {initial_loss:.2f}")

    # Configure training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        **TRAIN_ARGS
    )

    # Initialize Trainer with enhanced settings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[TrainingMonitorCallback()]
    )

    # Execute training
    try:
        print("\nStarting training...")
        trainer.train()
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print("Recommended actions:")
        print("1. Reduce learning rate (try 5e-6)")
        print("2. Check dataset formatting")
        print("3. Verify GPU memory allocation")
        exit(1)

    # Finalize training
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n🚀 Training complete! Model and Tokenizer saved to {OUTPUT_DIR}")
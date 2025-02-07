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
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "bf16": torch.cuda.is_available(),
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "fp16_full_eval": False,  # Disable mixed precision for validation
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
    
    def on_step_end(self, args, state, control, **kwargs):
        # Get model from kwargs instead of global scope
        model = kwargs.get('model')
        if model is None:
            return
        
        grads = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        
        # Check for empty gradients
        if not grads:
            print("Warning: No gradients found in this step")
            return
        
        # Check for gradient explosion
        max_grad = max(grads)
        if max_grad > 1e5:
            print(f"Gradient explosion detected: {max_grad:.2f}")

# Load and prepare dataset
def prepare_dataset():
    dataset = load_dataset(DATASET_NAME)
    
    # Handle dataset splits
    if isinstance(dataset, dict) and 'train' in dataset:
        return dataset['train'].train_test_split(test_size=0.25)
    return dataset.train_test_split(test_size=0.25)

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

# Modify tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    
    # Add this instead to verify tokenization:
    if len(tokenized["input_ids"]) == 0:
        raise ValueError("Empty tokenization result!")
    
    return tokenized

# Main execution
if __name__ == "__main__":
    # Prepare and tokenize dataset
    dataset = prepare_dataset()
    dataset = dataset.map(format_text, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # After tokenized_dataset = dataset.map(...)
    if len(tokenized_dataset["train"]) == 0:
        raise ValueError("Training dataset is empty after tokenization!")

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

    sample = tokenized_dataset["train"][0]
    print("Input IDs:", sample["input_ids"])
    print("Labels:", sample["input_ids"][1:] + [-100])  # Should be shifted
    print("Decoded Input:", tokenizer.decode(sample["input_ids"]))

    # Configure training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        logging_dir="./logs",
        report_to="tensorboard",
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
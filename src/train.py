import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import transformers  # For EarlyStoppingCallback

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "jhenberthf/filipino-gossip-dataset"
MAX_LENGTH = 1024
OUTPUT_DIR = "./marites_model"

# Training arguments with optimized settings
TRAIN_ARGS = {
    # Reduce memory usage and improve stability
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,  # Reduced from 16
    
    # Adjust learning rate and schedule
    "learning_rate": 1e-4,  # Increased slightly
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "linear",  # Changed to linear for stability
    "num_train_epochs": 3,
    
    # Memory optimization
    "gradient_checkpointing": True,
    "optim": "adamw_torch" if torch.cuda.is_available() else "adamw_torch",  # Use PyTorch's AdamW
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Reduce evaluation frequency to save memory
    "eval_strategy": "steps",
    "eval_steps": 300,
    "save_strategy": "steps",
    "save_steps": 300,
    "save_total_limit": 2,
    
    # Debug settings
    "logging_steps": 5,
    
    # Hardware-specific settings
    "fp16": False,
    "bf16": torch.cuda.is_available(),  # Use bfloat16 only with GPU
    "group_by_length": False,
    "report_to": ["tensorboard"],
}

class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.last_loss = float('inf')
        self.loss_increase_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            
            # Check for numerical instability
            if torch.isinf(torch.tensor(current_loss)) or torch.isnan(torch.tensor(current_loss)):
                print("\n⚠️ Numerical instability detected! Stopping training.")
                control.should_training_stop = True
                
            # Monitor loss spikes
            if current_loss > self.last_loss * 1.5:  # 50% increase
                self.loss_increase_count += 1
                if self.loss_increase_count >= 3:  # Three consecutive increases
                    print("\n⚠️ Multiple loss spikes detected! Consider reducing learning rate.")
            else:
                self.loss_increase_count = 0
                
            self.last_loss = current_loss
            
            # Monitor GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"\nGPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

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

class GradientMonitorCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        if model is None:
            return
        
        # Check gradients every 10 steps
        if self.step_count % 10 == 0:
            grad_norms = []
            zero_grad_params = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if param.grad is None:
                        zero_grad_params += 1
                    else:
                        grad_norms.append((name, param.grad.norm().item()))
            
            if zero_grad_params > 0:
                print(f"\nWarning: {zero_grad_params}/{total_params} parameters have no gradients")
            
            if grad_norms:
                max_grad_name, max_grad = max(grad_norms, key=lambda x: x[1])
                print(f"\nGradient stats - Max: {max_grad:.4f} ({max_grad_name})")
                
                if max_grad > 100:
                    print("⚠️ Large gradients detected! Consider reducing learning rate")
                elif max_grad < 1e-6:
                    print("⚠️ Very small gradients detected! Consider increasing learning rate")

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
    formatted = []
    for p, r in zip(examples['prompt'], examples['response']):
        # Check for None or empty strings
        if not p or not r or not isinstance(p, str) or not isinstance(r, str):
            continue
            
        try:
            formatted_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
                tokenize=False
            )
            formatted.append(formatted_text)
        except Exception as e:
            print(f"Warning: Failed to format example: {str(e)}")
            continue
            
    return {"text": formatted}

# Modify tokenization function
def tokenize_function(examples):
    try:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,  # Add padding
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        
        # Verify tokenization results
        if any(len(ids) == 0 for ids in tokenized["input_ids"]):
            print("Warning: Found empty sequences after tokenization")
            
        return tokenized
    except Exception as e:
        print(f"Tokenization error: {str(e)}")
        raise

# Add this function before main execution
def validate_dataset(dataset):
    """Validate dataset structure and content"""
    required_columns = ['prompt', 'response']
    
    for split in dataset:
        if not all(col in dataset[split].column_names for col in required_columns):
            raise ValueError(f"Missing required columns in {split} split")
            
        # Check for empty strings or None values
        for col in required_columns:
            empty_count = sum(1 for x in dataset[split][col] if not x or not isinstance(x, str))
            if empty_count > 0:
                print(f"Warning: Found {empty_count} empty/invalid entries in {split}/{col}")

# Add a learning rate finder callback
class LRFinderCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            current_loss = logs["eval_loss"]
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            # Adjust learning rate if loss plateaus
            if self.no_improvement_count >= 3:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.5
                print(f"\n📉 Loss plateaued. Reducing learning rate from {current_lr} to {new_lr}")
                
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                self.no_improvement_count = 0

# Add before trainer.train()
def validate_training_setup(trainer):
    print("\nValidating training setup...")
    
    # Test forward pass
    sample_batch = next(iter(trainer.get_train_dataloader()))
    try:
        outputs = trainer.model(**{k: v.to(trainer.model.device) for k, v in sample_batch.items()})
        print("✓ Forward pass successful")
        print(f"Initial loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        raise
    
    # Test backward pass
    try:
        outputs.loss.backward()
        print("✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    # Prepare and validate dataset
    dataset = prepare_dataset()
    validate_dataset(dataset)
    
    # Add early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01
    )
    
    # Prepare and tokenize dataset
    dataset = dataset.map(format_text, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    if len(tokenized_dataset["train"]) == 0:
        raise ValueError("Training dataset is empty after tokenization!")

    # Model initialization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # Enable gradients explicitly
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Verify gradients are enabled
    grad_params = sum(p.requires_grad for p in model.parameters())
    total_params = sum(1 for _ in model.parameters())
    print(f"Parameters requiring gradients: {grad_params}/{total_params}")

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
        logging_dir="./logs",
        load_best_model_at_end=True,
        **TRAIN_ARGS
    )

    # Initialize Trainer with enhanced settings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            TrainingMonitorCallback(),
            GradientMonitorCallback(),
            early_stopping
        ]
    )

    # Add before training starts
    validate_training_setup(trainer)

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
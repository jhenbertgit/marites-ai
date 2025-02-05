import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "jhenberthf/filipino-gossip-dataset"
MAX_LENGTH = 1024
OUTPUT_DIR = "./marites_model"
TRAIN_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "fp16": torch.cuda.is_available(),
    "optim": "adamw_torch",
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset(DATASET_NAME)
if "train" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
else:
    dataset = dataset["train"].train_test_split(test_size=0.1)

# Preprocessing function
def format_text(examples):
    texts = [
        f"Marites AI: {prompt}\nResponse: {response}{tokenizer.eos_token}"
        for prompt, response in zip(examples['prompt'], examples['response'])
    ]
    return {"text": texts}

# Apply formatting
dataset = dataset.map(
    format_text,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
        add_special_tokens=True,
    )

# Tokenize datasets
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Data collator (handles padding and labels creation)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    report_to="none",
    **TRAIN_ARGS
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# Train the model
if __name__ == "__main__":
    trainer.train()
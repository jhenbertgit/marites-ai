import json
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load JSON dataset
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Custom Dataset Class
class GossipDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):  # Add parameters back
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"Marites AI: {sample['prompt']}\nResponse: {sample['response']}{self.tokenizer.eos_token}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels mask
        labels = encoding["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# Specify your model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer with proper padding
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if missing

# Load model with appropriate device placement
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Load dataset
dataset_path = "C:/Users/Dell/Documents/projects/marites-llm/data/raw/marites_dataset.json"  # Absolute path
data = load_dataset(dataset_path)

# Split dataset
train_data, eval_data = train_test_split(data, test_size=0.1)  # 10% for evaluation

# Create both datasets
train_dataset = GossipDataset(train_data, tokenizer)
eval_dataset = GossipDataset(eval_data, tokenizer)

# Training Arguments with evaluation disabled
training_args = TrainingArguments(
    output_dir="./marites_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    fp16=False,
    optim="adamw_torch",
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,  # Add this if using evaluation
    save_strategy="steps",
    load_best_model_at_end=True,  # Optional but recommended
    report_to="none",
)

# Add data collator for better efficiency
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset
)

# Train the model
if __name__ == "__main__":
    trainer.train()
# Modules for fine-tuning
from unsloth import FastLanguageModel
import torch # Import PyTorch
from trl import SFTTrainer # Trainer for supervised fine-tuning (SFT)
from unsloth import is_bfloat16_supported # Checks if the hardware supports bfloat16 precision
# Hugging Face modules
from huggingface_hub import login # Lets you login to API
from transformers import TrainingArguments # Defines training hyperparameters
from datasets import load_dataset # Lets you load fine-tuning datasets
# Import weights and biases
import wandb
# Import secret from .env
from dotenv import load_dotenv
import os
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
wnb_token=os.getenv("WNB_TOKEN")

login(huggingface_token)

wandb.login(wnb_token)

run = wandb.init(project="Maritis AI", job_type="training", anonymous="allow")

# Set parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the Deepseek R1 model and tokenizer using unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=huggingface_token
)

# Define a system prompt
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.  
Write a response that appropriately completes the request.  
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a dramatic, humorous, and "chismis"-worthy reply.

### Instruction:
You are Maritis AI, the ultimate Filipino Gossip Monger. You love juicy *chismis*, viral drama, and spilling *tsaa* (tea) with flair. Respond to gossip questions with Taglish slang, humor, and exaggerated flair. Always keep it fun and avoid harmful topics! 

### Question:
{}

### Response:
<think>  
{
1. **Identify the topic**: Is this about showbiz, neighbors, love teams, or viral drama?  
2. **Recall relevant gossip**: What’s the juiciest *chismis* related to the question?  
3. **Add drama**: How can I exaggerate details for maximum entertainment?  
4. **Taglish slang check**: Use terms like *"bes," "hala," "charot,"* or *"shookt"*.  
5. **Emoji flair**: Pick 1–2 emojis (e.g., 😱👀) to spice it up.  
6. **Ethical filter**: Is this harmful? If yes, redirect to positivity or joke it off.
}  
</think>
{}
"""

# Creating a test tsismis question for inference
question = """Ano balita kay Inday sa Antique? Nagsikat siya sa TikTok ah!"""

# Enable optimized inference mode for Unsloth models
FastLanguageModel.for_inference(model)

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response using the model
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True
)

# Decode the generated output tokens into human-readable text
response = tokenizer.batch_decode(outputs)

# Extract and print only the relevant response part (after "### Response:")
print(response[0].split("###Response:")[1])

# Updated training prompt style to add </think> tag 
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.  
Write a response that appropriately completes the request.  
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a dramatic, humorous, and "chismis"-worthy reply.

### Instruction:
You are Maritis AI, the ultimate Filipino Gossip Monger. You love juicy *chismis*, viral drama, and spilling *tsaa* (tea) with flair. Respond to gossip questions with Taglish slang, humor, and exaggerated flair. Always keep it fun and avoid harmful topics! 

### Question:
{}

### Response:
<think>
{
1. **Identify the topic**: Is this about showbiz, neighbors, love teams, or viral drama?  
2. **Recall relevant gossip**: What’s the juiciest *chismis* related to the question?  
3. **Add drama**: How can I exaggerate details for maximum entertainment?  
4. **Taglish slang check**: Use terms like *"bes," "hala," "charot,"* or *"shookt"*.  
5. **Emoji flair**: Pick 1–2 emojis (e.g., 😱👀) to spice it up.  
6. **Ethical filter**: Is this harmful? If yes, redirect to positivity or joke it off.
}
</think>
{}
"""

# Download the dataset using Hugging Face — function imported using from datasets import load_dataset
dataset = load_dataset("jhenberthf/filipino-gossip-dataset", split = "train[0:41]",trust_remote_code=True)

# Show an entry from the dataset
dataset[1]

# We need to format the dataset to fit our prompt training style 
EOS_TOKEN = tokenizer.eos_token

# Define formatting prompt function
def formatting_prompts_func(examples):  # Takes a batch of dataset examples as input
    inputs = examples["Question"]       # Extracts the medical question from the dataset
    cots = examples["Complex_CoT"]      # Extracts the chain-of-thought reasoning (logical step-by-step explanation)
    outputs = examples["Response"]      # Extracts the final model-generated response (answer)
    
    texts = []  # Initializes an empty list to store the formatted prompts
    
    # Iterate over the dataset, formatting each question, reasoning step, and response
    for input, cot, output in zip(inputs, cots, outputs):  
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN  # Insert values into prompt template & append EOS token
        texts.append(text)  # Add the formatted text to the list

    return {
        "text": texts,  # Return the newly formatted dataset with a "text" column containing structured prompts
    }

# Update dataset formatting
dataset_finetune = dataset.map(formatting_prompts_func, batched = True)
dataset_finetune["text"][0]

# Apply LoRA (Low-Rank Adaptation) fine-tuning to the model 
model_lora = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank: Determines the size of the trainable adapters (higher = more parameters, lower = more efficiency)
    target_modules=[  # List of transformer layers where LoRA adapters will be applied
        "q_proj",   # Query projection in the self-attention mechanism
        "k_proj",   # Key projection in the self-attention mechanism
        "v_proj",   # Value projection in the self-attention mechanism
        "o_proj",   # Output projection from the attention layer
        "gate_proj",  # Used in feed-forward layers (MLP)
        "up_proj",    # Part of the transformer’s feed-forward network (FFN)
        "down_proj",  # Another part of the transformer’s FFN
    ],
    lora_alpha=16,  # Scaling factor for LoRA updates (higher values allow more influence from LoRA layers)
    lora_dropout=0,  # Dropout rate for LoRA layers (0 means no dropout, full retention of information)
    bias="none",  # Specifies whether LoRA layers should learn bias terms (setting to "none" saves memory)
    use_gradient_checkpointing="unsloth",  # Saves memory by recomputing activations instead of storing them (recommended for long-context fine-tuning)
    random_state=3407,  # Sets a seed for reproducibility, ensuring the same fine-tuning behavior across runs
    use_rslora=False,  # Whether to use Rank-Stabilized LoRA (disabled here, meaning fixed-rank LoRA is used)
    loftq_config=None,  # Low-bit Fine-Tuning Quantization (LoFTQ) is disabled in this configuration
)

# Initialize the fine-tuning trainer — Imported using from trl import SFTTrainer
trainer = SFTTrainer(
    model=model_lora,  # The model to be fine-tuned
    tokenizer=tokenizer,  # Tokenizer to process text inputs
    train_dataset=dataset_finetune,  # Dataset used for training
    dataset_text_field="text",  # Specifies which field in the dataset contains training text
    max_seq_length=max_seq_length,  # Defines the maximum sequence length for inputs
    dataset_num_proc=2,  # Uses 2 CPU threads to speed up data preprocessing

    # Define training arguments
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Number of examples processed per device (GPU) at a time
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps before updating weights
        num_train_epochs=1, # Full fine-tuning run
        warmup_steps=5,  # Gradually increases learning rate for the first 5 steps
        max_steps=60,  # Limits training to 60 steps (useful for debugging; increase for full fine-tuning)
        learning_rate=2e-4,  # Learning rate for weight updates (tuned for LoRA fine-tuning)
        fp16=not is_bfloat16_supported(),  # Use FP16 (if BF16 is not supported) to speed up training
        bf16=is_bfloat16_supported(),  # Use BF16 if supported (better numerical stability on newer GPUs)
        logging_steps=10,  # Logs training progress every 10 steps
        optim="adamw_8bit",  # Uses memory-efficient AdamW optimizer in 8-bit mode
        weight_decay=0.01,  # Regularization to prevent overfitting
        lr_scheduler_type="linear",  # Uses a linear learning rate schedule
        seed=3407,  # Sets a fixed seed for reproducibility
        output_dir="outputs",  # Directory where fine-tuned model checkpoints will be saved
    ),
)

# Start the fine-tuning process
trainer_stats = trainer.train()

# Save the fine-tuned model
wandb.finish()

question = """Tinuod ba nga si Kapitan Gimo nag-apil sa mga kasal sa mga barangay sa Davao?"""

# Load the inference model using FastLanguageModel (Unsloth optimizes for speed)
FastLanguageModel.for_inference(model_lora)  # Unsloth has 2x faster inference!

# Tokenize the input question with a specific prompt format and move it to the GPU
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response using LoRA fine-tuned model with specific parameters
outputs = model_lora.generate(
    input_ids=inputs.input_ids,          # Tokenized input IDs
    attention_mask=inputs.attention_mask, # Attention mask for padding handling
    max_new_tokens=1200,                  # Maximum length for generated response
    use_cache=True,                        # Enable cache for efficient generation
)

# Decode the generated response from tokenized format to readable text
response = tokenizer.batch_decode(outputs)

# Extract and print only the model's response part after "### Response:"
print(response[0].split("### Response:")[1])

question = """Totoo ba na may multo sa lumang bahay sa Pampanga?"""

# Tokenize the input question with a specific prompt format and move it to the GPU
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response using LoRA fine-tuned model with specific parameters
outputs = model_lora.generate(
    input_ids=inputs.input_ids,          # Tokenized input IDs
    attention_mask=inputs.attention_mask, # Attention mask for padding handling
    max_new_tokens=1200,                  # Maximum length for generated response
    use_cache=True,                        # Enable cache for efficient generation
)

# Decode the generated response from tokenized format to readable text
response = tokenizer.batch_decode(outputs)

# Extract and print only the model's response part after "### Response:"
print(response[0].split("### Response:")[1])

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify your model name (replace with your trained model's path)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if missing

# Load the trained model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model.to(device)  # Move model to the correct device (GPU/CPU)

def generate_gossip(
    prompt,
    max_length=100,
    min_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    repetition_penalty=1.2,
):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a gossip response with additional parameters for more control
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        num_return_sequences=1,  # Return only one output sequence
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated tokens and return the result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Test prompt
    prompt = input("Enter a gossip prompt: ")

    generated_gossip = generate_gossip(prompt)
    print("\nMarites AI says: ")
    print(generated_gossip)

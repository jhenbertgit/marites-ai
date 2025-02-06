import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify your model name
model_name = "./marites_model"

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if missing

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Move model to device
model.to(device)

# Function to generate response from the model
def generate_response(prompt, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Create attention mask

    # Generate text using the model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask, # Pass attention mask
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetitive phrases
        top_k=50,                # Top-k sampling
        top_p=0.9,              # Nucleus sampling
        temperature=0.8,         # Sampling temperature
        do_sample=True,          # Enable sampling instead of greedy decoding
    )

    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Marites AI is ready to chat! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break
        
        prompt = f"{user_input}\nResponse:"
        response = generate_response(prompt)
        print(f"Marites AI: {response}")
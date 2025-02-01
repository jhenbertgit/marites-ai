import os
import torch
from data import load_data, Tokenizer
from model.marites_model import MaritesModel
from torch.nn import functional as F

def generate(model, tokenizer, prompt, max_len=100, block_size=32):
    model.eval()
    # Encode the prompt and use only the last block_size tokens as context
    context = tokenizer.encode(prompt)[-block_size:]
    for _ in range(max_len):
        # Ensure the input is of shape (1, block_size)
        x = torch.tensor([context[-block_size:]]).long()
        logits = model(x)
        # Take the logits for the last token and apply softmax to obtain probabilities
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        context.append(next_id)
    return tokenizer.decode(context)

if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths to the required files
    data_path = os.path.join(current_dir, "..", "data", "raw", "input.txt")
    model_path = os.path.join(current_dir, "..", "outputs", "checkpoints", "marites_epoch_4000.pt")

    # Load model data and tokenizer from the input text
    text = load_data(data_path)
    tokenizer = Tokenizer(text)
    
    # Initialize the model using the same hyperparameters used during training
    model = MaritesModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=512,
        block_size=32,
        n_heads=8,
        n_layers=4
    )
    
    # Load the trained model state
    model.load_state_dict(torch.load(model_path))
    
    # Generate text using a custom prompt
    prompt = "Marites, may chismis ka ba?"  # <-- You can change the prompt here
    generated_text = generate(model, tokenizer, prompt)
    
    print(f"Prompt: {prompt}\nGenerated Text: {generated_text}")

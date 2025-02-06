import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper

# Custom logits processor to prevent numerical instability
class ClipLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Clip extreme values to prevent overflow/underflow
        scores = torch.clamp(scores, -50.0, 50.0)
        # Replace any remaining NaNs/Infs with finite values
        scores = torch.where(
            torch.isnan(scores) | torch.isinf(scores),
            torch.zeros_like(scores),
            scores
        )
        return scores

# Specify your model name
current_dir = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(current_dir, "..", "marites_model")

# Enhanced path validation
if not os.path.exists(model_name) or not os.path.isdir(model_name):
    raise ValueError(f"Invalid model directory: {model_name}")

# Device configuration with memory optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Load components with safety checks
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch_dtype,
)

# Ensure proper tokenizer configuration
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model loading verification
if not hasattr(model, "generate"):
    raise RuntimeError("Loaded model doesn't support generation")

# Move model to device and set evaluation mode
model.to(device)
model.eval()

def generate_response(prompt, max_length=150):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length // 2  # Leave room for generation
        ).to(device)

        # Pre-generation sanity check
        with torch.no_grad(), torch.autocast(device_type=device):
            test_output = model(**inputs)
            if torch.isnan(test_output.logits).any():
                raise RuntimeError("Model outputs contain NaN before generation")

        # Generation with stability enhancements
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            top_k=50,
            top_p=0.92,  # Slightly reduced from 0.95
            temperature=0.75,  # Adjusted for better stability
            repetition_penalty=1.15,  # Reduced from 1.2
            do_sample=True,
            logits_processor=[
                ClipLogitsProcessor(),  # Custom processor first
                TemperatureLogitsWarper(temperature=0.75),
                TopKLogitsWarper(top_k=50)
            ],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return response.split("Response:")[-1].strip()

    except RuntimeError as e:
        print(f"Generation error: {str(e)}")
        return "Sorry, I encountered an error processing that request."

if __name__ == "__main__":
    print("Marites AI is ready to chat! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat. Goodbye!")
            break
        
        if not user_input:
            print("Marites AI: Please say something meaningful!")
            continue
        
        prompt = f"{user_input}\nResponse:"
        response = generate_response(prompt)
        print(f"Marites AI: {response}")
        
        if device == "cuda":
            torch.cuda.empty_cache()
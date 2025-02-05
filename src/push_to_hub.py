from transformers import AutoModel, AutoTokenizer

# Load your model and tokenizer
model = AutoModel.from_pretrained("../marites_model/") # path of trained model directory
tokenizer = AutoTokenizer.from_pretrained("../marites_model/") # path of tokenizer of trained model directory

# Push to Hugging Face Hub
model.push_to_hub("jhenberthf/marites-ai")
tokenizer.push_to_hub("jhenberthf/marites-ai")

print('model and tokenizer pushed to huggingface')
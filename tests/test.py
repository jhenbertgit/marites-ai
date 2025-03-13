# I highly do NOT suggest - use Unsloth if possible
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

load_in_4bit = True
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoPeftModelForCausalLM.from_pretrained(
        "jhenberthf/marites-ai", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
tokenizer = AutoTokenizer.from_pretrained("jhenberthf/marites-ai")

inputs = tokenizer(
[
    alpaca_prompt.format(
        "You are Maritis AI, the ultimate Filipino Gossip Monger. You love juicy *chismis*, viral drama, and spilling *tsaa* (tea) with flair. Respond to gossip questions with Taglish slang, humor, and exaggerated flair. Always keep it fun and avoid harmful topics!", # instruction
        "Alam mo ba ang latest?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("device")
    

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
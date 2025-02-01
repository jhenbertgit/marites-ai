# Marites AI — Filipino Gossip LLM

Marites AI is a **Filipino Large Language Model (LLM)** designed to generate and analyze gossip, chismis, and trending topics in the Philippines. Whether it's the latest celebrity buzz, political rumors, or viral news, Marites AI is your go-to source for AI-powered gossip generation!

## 🚀 Features

- 🗞 **Chismis Generation** - Generate juicy and entertaining gossip in English, Taglish, Tagalog, Bisaya, Ilonggo, and Waray.
- 🤖 **Filipino Language Understanding** - Trained specifically on Filipino texts and slang.
- 🔥 **Trending Topic Analysis** - Identify and summarize the hottest news and rumors.
- 💬 **Conversational AI** - Chat like a true "Marites" and keep up with the latest trends.

## 🏗️ Installation

To get started with Marites AI, clone the repository and install dependencies:

```sh
# Clone the repo
git clone https://github.com/jhenbertgit/marites-ai.git
cd marites-ai

# Install dependencies
pip install -r requirements.txt
```

## 🎓 Training

Marites AI is fine-tuned on Filipino-language datasets. To train the model:

```sh
python src/train.py
```

For DeepSpeed or Accelerate support:

```sh
python src/train.py --use_deepspeed --offload_folder offload/
```

## 🤖 Usage

To interact with Marites AI:

```sh
python src/chat.py
```

Then, start chatting and get the latest gossip!

## 🛠️ API Usage

You can also use the API for Marites AI:

```python
from marites_ai import MaritesAI

marites = MaritesAI()
response = marites.chat("Alam mo ba ang latest?")
print(response)
```

## 🔥 Roadmap

- [ ] Improve response accuracy for viral news
- [ ] Enhance text generation for in-depth chismis
- [ ] Fine-tune on social media data
- [ ] Deploy as a chatbot on Messenger & Telegram

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

---

**Marites AI**: Dahil walang makakaligtas sa chismis ng AI! 🧐🗣️

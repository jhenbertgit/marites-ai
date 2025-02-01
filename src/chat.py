import argparse
import os
from marites_model import MaritesAI

def main():
    parser = argparse.ArgumentParser(description="Chat with Marites AI")
    parser.add_argument("--message", type=str, help="Message to send to Marites AI", required=False)
    args = parser.parse_args()

    model_path = "marites_model/marites_ai.pt"
    if not os.path.exists(model_path):
        print("Error: Marites AI model not found. Please run train.py first.")
        return

    marites = MaritesAI()
    
    if args.message:
        response = marites.chat(args.message)
        print(f"Marites AI: {response}")
    else:
        print("Welcome to Marites AI! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting Marites AI chat. Bye!")
                break
            response = marites.chat(user_input)
            print(f"Marites AI: {response}")

if __name__ == "__main__":
    main()
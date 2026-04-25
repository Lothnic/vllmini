"""CLI entry point."""
import torch
from transformers import AutoTokenizer
from models.weight_loader import load_hf_model
from engine.generator import Generator
from engine.sampler import Sampler

# CONFIG
HIDE_THINKING = False

# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_ID = "Qwen/Qwen3-1.7B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strip_thinking(output: str) -> str:
    if '</think>' in output:
        striped_output = output.split("</think>")[-1].strip()
        return striped_output
    else:
        return output

def main():
    model, config = load_hf_model(MODEL_ID, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    chat = [{"role": "user", "content": "Write a short story about a robot."}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # prompt = "Write a very long story about a robot."

    sampler = Sampler(temperature=0.7, top_p=0.9)
    gen = Generator(model, tokenizer, sampler)

    messages = []

    print(f"vLLMini Chat — Model: {MODEL_ID}")
    print("Commands: /exit, /reset, /history")
    print("-" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            break

        if user_input.lower() == "/reset":
            messages = []
            print("Chat reset.")
            continue

        if user_input.lower() == "/history":
            if not messages:
                print("No history.")
            else:
                for msg in messages:
                    print(f"{msg['role']}: {msg['content']}")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        full_output = gen.generate(prompt, max_new_tokens=2048)
        
        assistant_reply = full_output[len(prompt):].strip()
        if HIDE_THINKING:
            assistant_reply = strip_thinking(assistant_reply)
            print(f"Assistant: {assistant_reply}")
        else:
            print(f"Assistant: {assistant_reply}")
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
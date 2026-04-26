"""CLI entry point."""
import torch
from transformers import AutoTokenizer
from models.weight_loader import load_hf_model
from engine.generator import Generator
from engine.sampler import Sampler

# CONFIG
HIDE_THINKING = True  # Set to True to hide thinking blocks and show only final response

# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID = "Qwen/Qwen3-0.6B"
# MODEL_ID = "Qwen/Qwen3-1.7B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strip_thinking(output: str) -> str:
    if '</think>' in output:
        striped_output = output.split("</think>")[-1].strip()
        return striped_output
    else:
        return output

def main():
    model, config = load_hf_model(MODEL_ID, device=DEVICE)
    
    # Try to load tokenizer, fall back to local files only if offline
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
        print("Retrying with local_files_only=True...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
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
        
        parts = []
        buffer = ""           # accumulates raw text to detect tag boundaries
        thinking_done = False # flips True once we see </think>
        indicator_shown = False

        for token in gen.generate(prompt, max_new_tokens=2048):
            if HIDE_THINKING and not thinking_done:
                # Accumulate until we find the </think> closing tag
                buffer += token

                # Show a one-time indicator when we see <think>
                if not indicator_shown and "<think>" in buffer:
                    print("Thinking... ", end="", flush=True)
                    indicator_shown = True

                # Check if the thinking block has ended
                if "</think>" in buffer:
                    thinking_done = True
                    # Grab anything after </think> (model may emit response in same token)
                    remainder = buffer.split("</think>", 1)[1]
                    if remainder:
                        print(remainder, end="", flush=True)
                        parts.append(remainder)
                # Otherwise keep accumulating silently
            else:
                # Either HIDE_THINKING is False, or we're past </think>
                print(token, end="", flush=True)
                parts.append(token)

        print()
        assistant_reply = "".join(parts).strip()
        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
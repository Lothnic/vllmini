"""CLI entry point."""
import torch
from transformers import AutoTokenizer
from models.weight_loader import load_hf_model
from engine.generator import Generator
from engine.sampler import Sampler

# CONFIG
THINKING = False

# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    chat = [{"role": "user", "content": "Explain quantum computing in one sentence."}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    gen = Generator(model, tokenizer, Sampler(temperature=0.7, top_p=0.9))
    full_output = gen.generate(prompt, max_new_tokens=32*1000)
    
    print(f"Prompt: {prompt}")
    if THINKING==True:
        print(f"Output: {full_output}")
    else:
        print(strip_thinking(full_output))

if __name__ == "__main__":
    main()
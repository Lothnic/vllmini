"""CLI entry point."""
import torch
from transformers import AutoTokenizer
from models.weight_loader import load_hf_model
from engine.generator import Generator
from engine.sampler import Sampler

# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model, config = load_hf_model(MODEL_ID, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    chat = [{"role": "user", "content": "Explain quantum computing in one sentence."}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    gen = Generator(model, tokenizer, Sampler(temperature=0.7, top_p=0.9))
    output = gen.generate(prompt, max_new_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
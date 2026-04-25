"""vLLMini Benchmarking Harness"""
import torch
import time
import gc
from transformers import AutoTokenizer
from models.weight_loader import load_hf_model
from engine.sampler import Sampler
import argparse

# CONFIG
MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = "Explain the concept of backpropagation in deep learning in detail."
MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_separator():
    print("-" * 50)

@torch.inference_mode()
def benchmark(model_id:str, prompt:str, device:str):
    print(f"Loading model {model_id} to {device}...")
    model, config = load_hf_model(model_id, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampler = Sampler(temperature=0.0) # Greedy for consistency

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    print(f"Prompt length: {prompt_len} tokens")
    print("Starting warmup...")
    
    # Warmup
    model(input_ids[:, :5], position_ids=None)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Running benchmark...")
    print_separator()

    # --- PREFILL ---
    start_prefill = time.perf_counter()
    logits, past_key_values = model(input_ids, position_ids=None)
    next_token = sampler.sample(logits[:, -1, :])
    if device == 'cuda':
        torch.cuda.synchronize()
    end_prefill = time.perf_counter()
    
    ttft = (end_prefill - start_prefill) * 1000 # ms
    
    # --- DECODE ---
    decode_times = []
    generated_tokens = [next_token.item()]
    
    current_input_ids = next_token
    
    for _ in range(MAX_NEW_TOKENS - 1):
        start_decode = time.perf_counter()
        
        logits, past_key_values = model(
            current_input_ids, 
            position_ids=None, 
            past_key_values=past_key_values
        )
        next_token = sampler.sample(logits[:, -1, :])
        if device=='cuda':
            torch.cuda.synchronize()
        
        end_decode = time.perf_counter()
        decode_times.append(end_decode - start_decode)
        
        generated_tokens.append(next_token.item())
        current_input_ids = next_token
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    # --- STATS ---
    avg_itl = (sum(decode_times) / len(decode_times)) * 1000 if decode_times else 0
    total_gen_time = end_prefill - start_prefill + sum(decode_times)
    tokens_per_sec = len(generated_tokens) / total_gen_time
    if device=='cuda':
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
    else:
        peak_vram = 0

    print(f"Model ID: {model_id}")
    print(f"Tokens Generated: {len(generated_tokens)}")
    print(f"TTFT (Prefill): {ttft:.2f} ms")
    print(f"Avg ITL (Decode): {avg_itl:.2f} ms")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print_separator()
    
    # Output snippet to verify correctness
    print("Sample Output (first 50 chars):")
    print(tokenizer.decode(generated_tokens)[:100] + "...")

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLMini")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmark(args.model_id, args.prompt, args.device)

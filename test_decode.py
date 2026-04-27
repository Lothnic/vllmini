from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

tokens = tokenizer(" Hello world", return_tensors="pt").input_ids[0]
print(tokens)
print("Decode all:", repr(tokenizer.decode(tokens)))
print("Decode first token:", repr(tokenizer.decode(tokens[1:2])))
print("Decode second token:", repr(tokenizer.decode(tokens[2:3])))

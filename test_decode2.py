from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

tokens = tokenizer(" Hello world", return_tensors="pt").input_ids[0]
print(tokens)
t1 = tokenizer.decode(tokens[1:2])
t2 = tokenizer.decode(tokens[2:3])
print(repr(t1 + t2))

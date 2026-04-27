from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

tokens = tokenizer(" Hello world", return_tensors="pt").input_ids[0]
print(tokenizer.convert_ids_to_tokens(tokens))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

tokens = tokenizer(" Hello world", return_tensors="pt").input_ids[0]
print("tokens:", tokens)

prev_token_text = tokenizer.decode(tokens[1:2])
full_text = tokenizer.decode(tokens[1:3])
new_text = full_text[len(prev_token_text):]
print("prev_token_text:", repr(prev_token_text))
print("full_text:", repr(full_text))
print("new_text:", repr(new_text))

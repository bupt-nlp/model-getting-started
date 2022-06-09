import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "I am Good[SEP]KI"
tokens = tokenizer.tokenize(sentence)

result = tokenizer.encode(tokens, return_token_type_ids=True, return_tensors='pt', padding=True)
print(result)
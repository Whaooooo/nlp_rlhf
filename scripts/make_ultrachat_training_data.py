from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/zzo/Quickstart/asset/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(tokenizer.apply_chat_template(chat, tokenize=False))
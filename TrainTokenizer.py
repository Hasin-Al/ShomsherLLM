import os
from utils import read_text_file, add_eos, train_and_save_bpe_tokenizer

vocab_size = 10000

text_filepath = '/merged.txt'
text = read_text_file(text_filepath)
text_with_eos = add_eos(text)

# Ensure the directory exists
output_dir = "/tokenizer/bpe_tokenizer.json"
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

sentences = train_and_save_bpe_tokenizer(text_with_eos, vocab_size, output_dir)

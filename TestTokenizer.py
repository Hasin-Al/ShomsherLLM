from tokenizers import Tokenizer
from utils import read_text_file, train_and_save_bpe_tokenizer

bpe_tokenizer = Tokenizer.from_file('/tokenizer/bpe_tokenizer.json') # replace the file path with output_dir
vocab = bpe_tokenizer.get_vocab()

print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary sample: {list(vocab.items())[:10]}")

# 3️⃣ Encode some text
text = "বাংলা ভাষা খুব সুন্দর।<eos>"
encoding = bpe_tokenizer.encode(text)
print("Token IDs:", encoding.ids)
print("Tokens:", encoding.tokens)

# 4️⃣ Decode back to text
decoded_text = bpe_tokenizer.decode(encoding.ids)
print("Decoded text:", decoded_text)
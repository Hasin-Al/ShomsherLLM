<<<<<<< HEAD
from utils import read_text_file, add_eos, train_and_save_bpe_tokenizer


vocab_size = 10000

text_filepath = '/merged.txt'
text = read_text_file(text_filepath)
text_with_eos = add_eos(text)

output_dir = "/tokenizer/bpe_tokenizer.json" # change this directory link
=======
from utils import read_text_file, add_eos, train_and_save_bpe_tokenizer


vocab_size = 10000

text_filepath = '/merged.txt'
text = read_text_file(text_filepath)
text_with_eos = add_eos(text)

output_dir = "/tokenizer/bpe_tokenizer.json" # change this directory link
>>>>>>> b2a3c63f3e93c5bb09be358e3702f1e3bcc310d1
sentences= train_and_save_bpe_tokenizer(text_with_eos,vocab_size,output_dir)
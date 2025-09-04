
# Importing necessary libraries
import re
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers




# define a function to merge my txt files and clean them
def merge_and_clean(filepaths,output):
  bengali_pattern = re.compile(r"[^\u0980-\u09FF\u09E6-\u09EF\s.,!?;:()\"'—-]")
  with open(output,'w',encoding = 'utf-8') as output_file:
    for file in filepaths:
      with open(file,'r',encoding = 'utf-8') as inp_file:
        for line in inp_file:
          clean_line = bengali_pattern.sub("",line)
          clean_line = re.sub(r"\s+", " ", clean_line).strip()
          if clean_line:
            output_file.write(clean_line + "\n")

  return output

# function to add <eos> token at the end of each sentence
def add_eos(text):
  parts = re.split(r"([।!?])", text)
  final_sentences = []

  for i in range(0,len(parts)-1,2):
    sentence = parts[i].strip()
    ender = parts[i+1].strip()

    if sentence:
      final_sentences.append(sentence + ender+ '<eos>')

  return " ".join(final_sentences)

# function to read a text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n'," ")

    return text

# function to train our BPE tokenizer on our text data
def train_and_save_bpe_tokenizer(text,vocab_size,out_dir = 'tokenizer/bpe_tokenizer.json'):

    # separate sentences
    sentences = [s.strip() + "<eos>" for s in text.split("<eos>") if s.strip()]

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)


    #trainer
    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = ['<eos>']
    )

    #train
    tokenizer.train_from_iterator(sentences,trainer = trainer)

    # decoder
    tokenizer.decoder = decoders.ByteLevel()

    #save
    tokenizer.save(out_dir)
    print(f"tokenizer saved at {out_dir}")

    return sentences
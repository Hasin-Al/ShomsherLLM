import torch
from torch.utils.data import Dataset,DataLoader

#creating a dataset for my text.
class ShamDataset(Dataset):
  def __init__(self,text,tokenizer,max_len,stride):
    #defining input and targert
    self.input_ids = []
    self.target_ids = []

    #tokenize the text
    print(f"len of text is {len(text)}")
    tokens = tokenizer.encode(text).ids
    print(f"len of tokens is {len(tokens)}")
    print(f"max len is {max_len}")



    for i in range(0,len(tokens) - max_len,stride):
      inp_chunk = tokens[i:i+max_len]
      target_chunk = tokens[i+1:i+1+max_len]
      if len(inp_chunk) == max_len and len(target_chunk) == max_len:
        self.input_ids.append(torch.tensor(inp_chunk))
        self.target_ids.append(torch.tensor(target_chunk))


  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, index):
    return self.input_ids[index], self.target_ids[index]



def create_dataloader(text,tokenizer,max_len,stride,batch_size,shuflle = True, drop_last= True, worker = 0):

  dataset = ShamDataset(text,tokenizer,max_len,stride)
  dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = shuflle,drop_last = drop_last,num_workers = worker)

  return dataloader



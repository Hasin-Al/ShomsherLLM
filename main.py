import torch
from Dataset import create_dataloader
from utils import read_file,add_eos
from tokenizers import Tokenizer
from model import ShomsherLLM
import torch.optim as optim
from train import train_model

# Dataset createion and dataloader
#define parameters
stride = 5
batch = 10
val_batch = 4
max_len = 256

# load tokenizer
bpe_tokenizer = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")

# read the merged and cleaned text file
text = read_file("/merged.txt")   # you add your cleaned data path here
text_with_eos = add_eos(text)

# split the text into train and val
train_text = text_with_eos[:int(0.9*len(text_with_eos))]   # first 90%
val_text   = text_with_eos[int(0.9*len(text_with_eos)):]   # last 10%

# create dataloader
train_dataloader = create_dataloader(train_text,bpe_tokenizer,max_len,stride,batch)
val_dataloader = create_dataloader(val_text,bpe_tokenizer,max_len,stride,val_batch,shuflle= False)
d_iter = iter(train_dataloader)
first_batch = next(d_iter)
#print(first_batch)
print(f"train dataset size {len(train_dataloader)}")
print(f"val dataset size {len(val_dataloader)}")

# Defining our model here 
# configuration file
CONFIG = {
"vocab_size": 5515,
"context_length": 256,
"emb_dim": 200,
"num_heads": 8,
"num_layers": 6,
"dropout": 0.1,
"rope_dim":24,
"compressed_dim": 50,
"bias_qkv": False
}
torch.manual_seed(123)
model = ShomsherLLM(CONFIG)
#print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Learning rate, optimizer and device
learning_rate = 5e-5    # start small for large models
weight_decay = 0.01     # standard value for LLMs
betas = (0.9, 0.999)    # Adam defaults, usually fine
eps = 1e-8              # numerical stability
optimizer = optim.AdamW(model.parameters(),
                        lr=learning_rate,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)

# Training the model here
tr_loss,val_loss = train_model(model,train_dataloader,val_dataloader,optimizer,num_epochs=1,eval_freq=1000,device = 'cuda')

# Save the model
# Path to save
save_path = "model/ShomsherLLM_checkpoint.pth"

# Save model and optimizer state_dicts along with epoch and step info
torch.save({
 # last completed epoch
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
     # optional: final validation loss
}, save_path)

print(f"Model and optimizer saved to {save_path}")



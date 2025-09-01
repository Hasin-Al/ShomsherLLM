import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import ShomsherLLM
import torch.optim as optim
# Text generation function using top-k sampling and temperature scaling
def generate_text(model, tokenizer, prompt, max_length=100, top_k=50, temperature=1.0, device='cuda'):

    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt).ids, device=device).unsqueeze(0)  # Shape: [1, seq_len]

    generated = input_ids.tolist()[0]  # Keep track of generated token ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # Take the last token's logits

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from the distribution
            next_token = top_k_indices[0, torch.multinomial(probs[0], 1)].item()

            generated.append(next_token)

            # Append next token to input_ids
            input_ids = torch.tensor([generated], device=device)

    # Decode token ids to string
    text = tokenizer.decode(generated)
    return text

# Text generation function using KV cache for faster inference
#generate text using KV caching
def generate_text(model, tokenizer, prompt, max_length=100, top_k=50, temperature=1.0, device='cuda'):
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)  # [1, seq_len]
    generated = input_ids.tolist()[0]

    past_kv = None  # Initialize cache
    position_offset = 0  # absolute token position for RoPE

    with torch.no_grad():
        for _ in range(max_length):
            # --- first step: feed full prompt ---
            if past_kv is None:
                logits, past_kv = model(
                    input_ids,
                    return_cache=True,
                    seq_start=position_offset   # start from 0
                )
                position_offset += input_ids.size(1)

            # --- subsequent steps: feed last token only ---
            else:
                last_token = input_ids[:, -1:]  # shape [1,1]
                logits, past_kv = model(
                    last_token,
                    return_cache=True,
                    past_kv=past_kv,
                    seq_start=position_offset  # absolute position
                )
                position_offset += 1

            # --- take logits of last token ---
            next_token_logits = logits[:, -1, :] / temperature

            # --- top-k sampling ---
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs[0], 1)].item()

            # --- append to generated sequence ---
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], device=device)  # feed only last token next

    # Decode generated token ids
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text



 
# Example Bengali prompt
bengali_prompt = "আজকে সকালে"

# Generate text using the trained model
# Load tokenizer and model
bpe_tokenizer = Tokenizer.from_file('/content/bpe_tokenizer.json')
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
model = ShomsherLLM(CONFIG)

# Recreate the optimizer with the SAME hyperparams
learning_rate = 5e-5
weight_decay = 0.01
betas = (0.9, 0.999)
eps = 1e-8
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
)

# Load checkpoint
checkpoint = torch.load("model/ShomsherLLM_checkpoint.pth", map_location="cpu")
# Restore model + optimizer states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("✅ Model & optimizer restored successfully!")

# Ready for inference
model.eval()

# Or resume training
# model.train()

generated_text = generate_text(
    model=model,
    tokenizer=bpe_tokenizer,
    prompt=bengali_prompt,
    max_length=100,   # number of tokens to generate
    top_k=25,         # top-k sampling
    temperature=0.1,  # temperature scaling
    device='cuda'
)

print("Generated Bengali Text:\n", generated_text)

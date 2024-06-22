# AI -> Andrej's Insight

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import time 

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open(f'input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text[:2000])
        self.tokens = torch.tensor(tokens)
        print(f'1 epochs =  {len(self.tokens) // (B*T)} batches')
    
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Improve this later, does the work though
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
            return None, None

        buf = self.tokens[self.current_position: self.current_position + (B*T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T

        
        return x, y
        

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module): # AI: can be seen as a map reduce kind of operation
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config) # AI: Attention is a aggregation/pooling/reduction operation (~ reduce)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config) # AI: This happens for each tokens individually (~ map)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        """Weight Sharing
        - The 2 embedding weights and the pre-softmax linear transformation 
        share the same weight.
        - They have the same pointer for the memory
        - This has been mentioned in the `Attention Is All You Need` paper
        - It references to some paper published from `Tel Aiviv`
        """

        # weight sharing scheme
        # This acrually reduces VRAM in my case (7900xtx) by 2%
        # without sharing - 18%
        # with sharing - 16%
        self.transformer.wte.weight = self.lm_head.weight

        # weight initilization
        # self.apply() is a function of nn.Module that applies a function
        # on all the submodules

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal(module.weight, mean = 0.0, std = 0.02)
        

    def forward(self, idx, label = None):
        # idx is of shape B, T
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device= idx.device)
        pos_embed = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_embed = self.transformer.wte(idx) # token embedding of shape (B, T, n_embed)
        x = tok_embed + pos_embed

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if label is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 

num_response_sequence = 5
max_length = 30

# Auto detect CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

## HF weights loading 
# model = GPT.from_pretrained('gpt2')
# print("didn't crash !!!")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# for training (random initialization))
model = GPT(GPTConfig())
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_response_sequence, 1)
# x = tokens.to(device)

# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# x = buf[:B*T].view(B, T).to(device)
# y = buf[1:].view(B, T).to(device)

# # Testing loss for initialization
# logits, loss = model(x, y)
# print(f"logits shape: {logits.shape}")
# print(f"loss: {loss.item()}")

trainloader = DataLoaderLite(B=4, T=32)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    epoch_loss = []
    while True:
        x, y = trainloader.next_batch()
        if x is None and y is None: # epoch done!!
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        import code; code.interact(local=locals())
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
        # print(f"step: {i}, loss: {loss.item()}")
    
    print(f'epoch {i} loss -> {sum(epoch_loss) / len(epoch_loss)}')



 # # generate
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) <= max_length:
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:, -1, :] # (B, vocab)
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (5, 50)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         x = torch.cat((x, xcol), dim=1)

# for i in range(num_response_sequence):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)







    
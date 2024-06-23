# Using `tinygrad`

from tinygrad import Tensor, nn, dtypes
from dataclasses import dataclass
import math
import numpy as np
import time

MODEL_TYPE_LIST = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
TRANSPOSE_LAYER = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GELUWRAPPER:
    def __init__(self) -> None:
        pass
    def forward(self, x):
        return x.gelu()



class MLP:
    def __init__(self, config) -> None:
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = GELUWRAPPER()
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
    
    def forward(self, x):
        _ = self.c_fc(x)
        _ = self.gelu.forward(_)
        _ = self.c_proj(_)

        return _
    

class CausalSelfAttention:
    def __init__(self, config):
        
        # check for multi-head compatibility
        assert config.n_embed % config.n_head == 0

        # multiply by 3 for k,q,v
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        tmp = qkv.shape[-1] // 3
        q, k, v = qkv[:, :, :tmp], qkv[:, :, tmp:2*tmp], qkv[:, :, 2*tmp:3*tmp]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = q.scaled_dot_product_attention(k, v, is_causal = True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y
    

class Block:
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x += self.attn.forward(self.ln_1(x))
        x += self.mlp.forward(self.ln_2(x))

        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class GPT:
    def __init__(self, config):
        self.config = config

        # Create the transformer structure
        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.wpe = nn.Embedding(self.config.block_size, self.config.n_embed)

        self.h = []

        for _ in range(self.config.n_layer):
            self.h.append(
                Block(config)
            )
        
        self.ln_f=  nn.LayerNorm(self.config.n_embed)
        # End of the transformer structure

        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size)

        # weight sharing
        

    def forward(self, x, label = None):
        B, T = x.shape

        assert T <= self.config.block_size, f"cannot forward seq len {T} and block size {self.config.block_size}"

        pos = Tensor.arange(T, dtype=dtypes.long)
        pos_embed = self.wpe(pos)
        tok_embed = self.wte(x)
        x = pos_embed.add(tok_embed)

        for block in self.h:
            x = block.forward(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None

        if label is not None:
            # cal the loss
            loss = logits.view(B*T, logits.shape[-1]).sparse_categorical_crossentropy(label.view(B*T)).item()
        
        return logits, loss


num_response_sequence = 5
max_length = 30

# Random initialization
model = GPT(GPTConfig())

print(model)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = Tensor([tokens for _ in range(5)])

print(type(tokens), tokens.shape)

# Random Model Generation
B, T = tokens.shape
while tokens.shape[-1] <= max_length:
    # print(tokens.shape)
    s = time.time()
    logits, loss = model.forward(tokens)
    e = time.time()
    print(f'inf time: {e - s}sec')
    logits = logits[:, -1, :] # (B, T)
    probs = logits.softmax()
    idx = probs.argmax(axis = 1).view(B, 1)
    # print(tokens.numpy().shape, idx.numpy().shape)
    tokens = Tensor(np.concatenate([tokens.numpy(), idx.numpy()], axis = 1))


for i in range(num_response_sequence):
    sent = tokens[i, :max_length].numpy().tolist()
    decoded = enc.decode(sent)
    print(">", decoded)


        

        
        
        



        

    



#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
from typing import Optional, Union
import argparse
from tqdm import trange
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Timing, DEBUG, getenv, fetch, colored
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from tinygrad.shape.symbolic import Variable

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)
HALF = getenv("HALF")

tokens = open('tokens.txt', 'r').readlines()
token_dict = dict()
max_token_length = -1
for i in range(len(tokens)): 
  s = tokens[i].replace("\n","").replace("/n","\n")
  token_dict[s] = i
  if len(s) > max_token_length:
    max_token_length = len(s)
def rory_decode(index):
  ret = ""
  for i in index:
    ret+=tokens[i].replace("\n","").replace("/n","\n") #hack with linebreak
  return ret

#rory can we match linear??
#start using numpy?
class Rory_Linear():
  def __init__(self,weight=None):
    self.weight = weight

  def __call__(self,x):
    return self.weight.matmul(x)

Tensor.manual_seed(420)
input = Tensor.rand([10])
tg_linear = Linear(10,10,bias=False)
out = tg_linear(input)
print(out.numpy())
out = tg_linear.weight.matmul(input)
print(out.numpy())
rory_linear = Rory_Linear(tg_linear.weight)
out = rory_linear(input)
print(out.numpy())

def rory_lm_head():
  return None

def rory_encode(x):
  ret = []
  token = None
  i = -1
  while len(x) > 0:
    token = None
    i = -1
    while token == None:
      i+=1
      s = x[:min(max_token_length,len(x))-i]
      #print("s =",s)
      if s in token_dict:
        token = token_dict[s]
    ret.append(token)
    x = x[min(max_token_length,len(x))-i:]
  return ret

class Attention:
  def __init__(self, dim, n_heads):
    self.c_attn = Linear(dim, 3*dim, bias=True)
    self.c_proj = Linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]) -> Tensor:
    if mask is not None or start_pos.val == 0:
      # no symbolic shape qkv when consuming prompts
      start_pos = start_pos.val

    if HALF: x = x.half()
    xqkv = self.c_attn(x)
    xq, xk, xv = [xqkv.shrink((None, None, (i*self.dim, (i+1)*self.dim))).reshape(None, None, self.n_heads, self.head_dim) for i in range(3)]
    bsz, seqlen, _, _ = xq.shape

    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype)

    if start_pos > 0:
      keys = self.cache_kv[0].shrink((None, (0, start_pos), None, None)).cat(xk, dim=1)
      values = self.cache_kv[1].shrink((None, (0, start_pos), None, None)).cat(xv, dim=1)
    else:
      keys = xk
      values = xv

    # update the cache
    new_cache = Tensor.stack([keys, values]).pad((None, None,(0,MAX_CONTEXT-start_pos-seqlen),None,None)).contiguous()
    self.cache_kv.assign(new_cache).realize()

    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    return self.c_proj(xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, self.dim))

class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.c_proj(self.c_fc(x).gelu())

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.mlp = FeedForward(dim, 4*dim)
    self.ln_1 = LayerNorm(dim, norm_eps)
    self.ln_2 = LayerNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]):
    h = x + self.attn(self.ln_1(x), start_pos, mask).float()
    return (h + self.mlp(self.ln_2(h)))

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)
    self.rory_lm_head = Rory_Linear(None) #fix late
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Union[Tensor,Variable], start_pos:Variable, temperature:float=0.0):
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
    if isinstance(tokens, Variable):
      seqlen = 1
      tok_emb = self.wte.weight.shrink(((tokens, tokens+1), None))
    else:
      seqlen = tokens.shape[1]
      tok_emb = self.wte(tokens)

    pos_emb = self.wpe(self.allpos.shrink((None, (start_pos, start_pos+seqlen))))
    h = tok_emb + pos_emb

    if HALF: h = h.half()

    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None

    for hi in self.h: h = hi(h, start_pos, mask)

    #print("rory lm_head in =",type(self.ln_f(h)),self.ln_f(h).shape)
    #print("rory lm_head bias =",self.lm_head.bias)
    #print("rory lm_head weights =",self.lm_head.weight.shape)
    logits = self.lm_head(self.ln_f(h))

    if logits.shape[1] == 0:
      # special case for empty prompt
      logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
    else:
      logits = logits[:, -1, :]

    if temperature < 1e-6:
      ret = logits.argmax(-1)
    else:
      ret = (logits / temperature).softmax().multinomial()
    return ret.flatten().realize()

  def __call__(self, tokens:Tensor, start_pos:Variable, temperature:float=0.0) -> Tensor:
    forward = (self.forward_jit if (isinstance(tokens, Variable) or tokens.shape[1] == 1) and getenv("JIT") else self.forward)
    return forward(tokens, start_pos, temperature)

VOCAB_SIZE = 50257
class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    model = Transformer(n_layers=12,n_heads=12,dim=768,norm_eps=1e-5,vocab_size=VOCAB_SIZE) #small
    weights = torch_load(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'))
    # special treatment for the Conv1D weights we need to transpose
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    weights['lm_head.weight'] = weights['wte.weight']
    weights['rory_lm_head.weight'] = weights['wte.weight']
    load_state_dict(model, weights)

    return GPT2(model)

  def __init__(self, model):
    self.model = model

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    prompt_tokens = rory_encode(prompt)
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      if batch_size == 1 and len(toks[0][start_pos:]) == 1:
        tokens = Variable("tokens", 0, VOCAB_SIZE).bind(toks[0][start_pos])
      else:
        tokens = Tensor([x[start_pos:] for x in toks])
      tok = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), temperature).numpy().tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    return [rory_decode(x) for x in toks]

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")
  default_prompt = "What is the answer to life, the universe, and everything?"

  Tensor.manual_seed(420)
  np.random.seed(420)

  gpt2 = GPT2.build()

  texts = gpt2.generate(prompt=default_prompt, max_length=100, temperature=0.8, timing=None, batch_size=1)
  print('Generating text...')
  for i,text in enumerate(texts): print(colored(f"Response {i}:", "green"), text)

  assert texts == [("What is the answer to life, the universe, and everything?"
  "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
  "be a bit different. For one, you might take a longer view of the universe. "
  "But for other people — including scientists, artists, and other in-your-face "
  "people — your answer might be far more like: Life doesn't exist at all.\n\n"
  "Imagine you are a young person who just graduated from middle school and has "
  "never really pursued a career in astrophysics. You're on an eight")]
  exit()
#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Optional, Union, Tuple
from tqdm import trange
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv, fetch, colored, all_int
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import dtypes
import math
rorys = True #todo args
import inspect

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)

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

def rory_scaled_dot_product_attention(x, key, value, attn_mask:Optional[Tensor]=None,
                                  dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
  my_mask = np.triu(np.full([np.shape(x)[2],np.shape(x)[2]],1)) 
  my_mask = (my_mask - np.eye(np.shape(x)[2])) * -math.inf
  my_mask[np.isnan(my_mask)] = 0
  np.where(np.isnan(my_mask), 0, my_mask) # inf * 0 = nan
  my_mask = [[my_mask]]
  key = np.transpose(key,(0,1,3,2))
  qk = np.matmul(x,key)
  qk = qk / math.sqrt(np.shape(x)[-1])
  qk = qk + my_mask
  for a in range(len(qk)):
    for b in range(len(qk[0])):
      for c in range(len(qk[0][0])):
        qk[a][b][c] = np.exp(qk[a][b][c] - np.max(qk[a][b][c]))
        qk[a][b][c] = qk[a][b][c] / qk[a][b][c].sum()
  qk = np.matmul(qk,value)
  return qk

#rory can we match linear??
#start using numpy?
class Rory_Linear():
  def __init__(self, in_features, out_features, bias=True):
    # TODO: is this init good? torch inits to uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    bound = 1 / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self,x):
    #rory this is terrible atm obv
    w = self.weight.numpy()
    if self.bias:
      b = self.bias.numpy()
    w = w.transpose()
    x = x[0]
    ret = np.matmul(x,w)
    if self.bias:
      for x in range(ret.shape[0]):
        ret[x] += b
    ret = [ret]
    return ret
  
class Rory_LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape)) if elementwise_affine else (None, None)

  def __call__(self, x):
    w,b = self.weight.numpy(),self.bias.numpy()
    if np.shape(x)[1] == 1:
      x = x[0][0]
      x = (x - x.mean()) / np.sqrt(np.mean((x - x.mean())**2) + self.eps)\
      * w + b
      x = [[x]]
      return x
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    #x = x.layernorm(eps=self.eps, axis=self.axis)
    #if not self.elementwise_affine: return x
    #return x * self.weight + self.bias
    #it doesnt work still with actual copy?
    for i in range(len(x[0])):
      x[0][i] = (x[0][i] - x[0][i].mean()) / np.sqrt(np.mean((x[0][i] - x[0][i].mean())**2) + self.eps)\
      * w + b
    return x

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
      if s in token_dict:
        token = token_dict[s]
    ret.append(token)
    x = x[min(max_token_length,len(x))-i:]
  return ret

class Attention:
  def __init__(self, dim, n_heads):
    self = self

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]) -> Tensor:
    return None

class Rory_Attention:
  def __init__(self, dim, n_heads):
    self.c_attn = Rory_Linear(dim, 3*dim, bias=True)
    self.c_proj = Rory_Linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos:Variable, mask:Optional[Tensor]) -> Tensor:
    if mask is not None or start_pos.val == 0:
      # no symbolic shape qkv when consuming prompts
      start_pos = start_pos.val

    xqkv = self.c_attn(x)

    # rory this is bad now obv
    xq = np.zeros(shape=(1,np.shape(xqkv)[1],self.dim))
    for i in range(xq.shape[1]):
      xq[0][i] = xqkv[0][i][0:self.dim]
    xq = xq.reshape(1,xq.shape[1],self.n_heads,self.head_dim)
    xk = np.zeros(shape=(1,np.shape(xqkv)[1],self.dim))
    for i in range(xk.shape[1]):
      xk[0][i] = xqkv[0][i][self.dim:2*self.dim]
    xk = xk.reshape(1,xk.shape[1],self.n_heads,self.head_dim)
    xv = np.zeros(shape=(1,np.shape(xqkv)[1],self.dim))
    for i in range(xv.shape[1]):
      xv[0][i] = xqkv[0][i][self.dim*2:3*self.dim]
    xv = xv.reshape(1,xv.shape[1],self.n_heads,self.head_dim)
    bsz, seqlen, _, _ = xq.shape
    
    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = np.zeros(shape=[2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim])
      self.cache_kv = Tensor(self.cache_kv)

    if start_pos > 0:
      keys = self.cache_kv[0]
      values = self.cache_kv[1]
      keys_np = keys.numpy()
      values_np = values.numpy()
      ret = [keys_np,values_np]
      #terrible loop
      for a in range(len(ret)):
          for b in range(len(ret[0])):
              for c in range(start_pos.unbind()[1]+1,len(ret[0][0])):
                  ret[a][b][c] = np.zeros_like(ret[a][b][c])
          if start_pos.unbind()[1] > -1 and start_pos.unbind()[1] < len(ret[0][0]):
              ret[0][b][start_pos.unbind()[1]] = xk[0][0]
              ret[1][b][start_pos.unbind()[1]] = xv[0][0]
      new_cache = Tensor(ret)
      self.cache_kv.assign(new_cache).realize()
      new_cache = None #todo not needed?
      
      xq = xq.transpose((0,2,1,3)) # same as (1,2) in tinygrad

      #todo below 
      # start pos = start_pos[1-MAX_CONTENT=pos] so [1-128=13] ...[1-128=111]
      # keys and values both shape (1,128,12,64)
      # xq shape is (1,128,1,64)
      # xk and xv shape is (1, 1, 12, 64)
      # .... can numpy below not with start_pos.unbind()[1]

      keys = keys.numpy()
      values = values.numpy()
      s = list(np.shape(keys))
      s[1] = start_pos.unbind()[1]
      keys_small = np.empty(s)
      values_small = np.empty(s)
      for i in range(len(keys_small[0])):
        keys_small[0][i] = keys[0][i]
        values_small[0][i] = values[0][i]
      keys = keys_small
      values = values_small
      keys = np.concatenate([keys,xk],axis=1)
      values = np.concatenate([values,xv],1)
      keys, values = keys.transpose(0,2,1,3), values.transpose(0,2,1,3)
      keys = keys.transpose(0,1,3,2)
      qk2 = np.matmul(xq,keys)

      qk2 = qk2 / math.sqrt(xq.shape[-1])
      for a in range(len(qk2[0])):
        for b in range(len(qk2[0][a])):
          qk2[0][a][b] = np.exp(qk2[0][a][b]  - np.max(qk2[0][a][b] ))
          qk2[0][a][b]  = qk2[0][a][b]  / qk2[0][a][b] .sum()
      qk2 = np.matmul(qk2,values)
      xq = qk2 
      xq = xq.transpose((0,2,1,3))
      xq = xq.reshape((bsz,seqlen,self.dim))
      ret = self.c_proj(xq)
      return ret

    keys = xk
    values = xv
    values_np = values
    s = list(np.shape(keys))
    s[1] = MAX_CONTEXT
    new_cache = np.zeros(shape=s)
    new_cache = [np.copy(new_cache),np.copy(new_cache)]
    for i in range(len(keys[0])):
      new_cache[0][0][i] = keys[0][i]
      new_cache[1][0][i] = values_np[0][i]       
    new_cache = Tensor(new_cache)
    self.cache_kv.assign(new_cache).realize()
    xq = xq.transpose((0,2,1,3)) # same as (1,2) in tinygrad
    #can't numpy them outside this if!
    keys, values = keys.transpose((0,2,1,3)), values.transpose((0,2,1,3))
    
    xq = rory_scaled_dot_product_attention(xq,keys,values,mask)
    xq = xq.transpose((0,2,1,3))
    #xq = xq.transpose(1, 2)
    xq = xq.reshape(bsz, seqlen, self.dim) #todo !
    ret = self.c_proj(xq)
    return ret
  

class FeedForward:
  def __init__(self, dim, hidden_dim):
    self = self

class Rory_FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Rory_Linear(dim, hidden_dim, bias=True)
    self.c_proj = Rory_Linear(hidden_dim, dim, bias=True)

  def __call__(self, x) -> Tensor:
    x = self.c_fc(x)
    for i in range(np.shape(x)[1]):
      # gelu() activation
      x[0][i] = 0.5 * x[0][i] * (1 + np.tanh(x[0][i] * 0.7978845608 * (1 + 0.044715 * x[0][i] * x[0][i])))
    ret = self.c_proj(x)
    return ret
  
class Rory_Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = Tensor.zeros(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'vocab_counter'):
      self.vocab_counter = [[np.arange(start=0,stop=self.vocab_size)]]
    batch_size, seqlen = idx.shape
    if seqlen == 0:
      print("rory seq len is 0")
      exit() 
      return Tensor.empty(batch_size, 0, self.embed_size, device=self.weight.device)

    if idx.shape[1] == 1:
      b = np.repeat(False,self.vocab_size)
      b[idx] = True
      w = self.weight.numpy()
      ret = [[np.matmul(b,w)]]
      return ret
    
    b = np.empty((1,idx.shape[1],self.vocab_size),dtype=bool)
    b.fill(False)
    w = self.weight.numpy()
    for i in range(len(b[0])):
      b[0][i][i] = True
    ret = np.matmul(b,w)
    return ret
  
class Rory_Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = Tensor.zeros(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    w = self.weight.numpy()
    idx = idx.numpy()
    if not hasattr(self, 'vocab_counter'):
      self.vocab_counter = np.arange(self.vocab_size)
      self.vocab_counter = self.vocab_counter.reshape(1,1,self.vocab_size)
    idx_np = []
    for i in range(len(idx[0])):
      idx_np.append([idx[0][i]])
    idx_np = ([idx_np] == self.vocab_counter)
    ret = np.matmul(idx_np,w)
    return ret


class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.rory_attn = Rory_Attention(dim,n_heads)
    self.mlp = FeedForward(dim, 4*dim)
    self.rory_mlp = Rory_FeedForward(dim, 4*dim)
    self.ln_1 = LayerNorm(dim, norm_eps) #partly done
    self.rory_ln_1 = Rory_LayerNorm(dim,norm_eps)
    self.ln_2 = LayerNorm(dim, norm_eps) #done
    self.rory_ln_2 = Rory_LayerNorm(dim,norm_eps)

  def __call__(self, x, start_pos:Variable, mask:Optional[Tensor],np_in=False):
    if np_in:
      h = np.copy(x)
      ln1 = self.rory_ln_1(x)
      attn = self.rory_attn(ln1,start_pos,mask)
      h += attn
      h2 = np.copy(h)
      ln2 = self.rory_ln_2(h2) 
      mlp = self.rory_mlp(ln2)
      #h = Tensor(h)
      #mlp = Tensor(mlp)
      ret = mlp + h
      return ret
    exit()
    h = x
    ln1 = self.rory_ln_1(x.numpy())
    attn = self.rory_attn(ln1,start_pos,mask)
    h += Tensor(attn)
    ln2 = self.rory_ln_2(h.numpy())
    mlp = Tensor(self.rory_mlp(ln2))
    return (h + mlp)

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding(vocab_size, dim)
    self.rory_wte = Rory_Embedding_2(vocab_size,dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.rory_wpe = Rory_Embedding(max_seq_len,dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.rory_ln_f = Rory_LayerNorm(dim,norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)
    self.rory_lm_head = Rory_Linear(dim, vocab_size, bias=False) #fix late
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Union[Tensor,Variable], start_pos:Variable, temperature:float=0.0):
    if not hasattr(self, 'allpos'): 
      self.allpos = np.arange(0, MAX_CONTEXT).reshape(1, -1)
    if isinstance(tokens, Variable):
      seqlen = 1
      tok_emb = self.rory_wte.weight
      tok_emb = tok_emb.numpy()
      tok_emb = tok_emb[tokens.unbind()[1]] # same as tok_emb.shrink(((tokens, tokens+1), None))
    else:
      seqlen = tokens.shape[1]
      tok_emb = self.rory_wte(tokens) #rorys todo

    s = list(np.shape(self.allpos))
    s[1] = seqlen
    allpos_s = np.empty(s,dtype=np.int32)
    for i in range(seqlen):
      allpos_s[0][i] = self.allpos[0][start_pos.unbind()[1] + i]
    pos_emb = self.rory_wpe(allpos_s)
    h = Tensor(tok_emb) + Tensor(pos_emb)

    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None

    #rory - h self.h is the 12 transformer blocks, so this is just forward through all
    for hi in self.h:
      if type(h) == Tensor:
        h = hi(h.numpy(), start_pos, mask,np_in=True)
      else:
        h = hi(h, start_pos, mask,np_in=True)

    h = self.rory_ln_f(h)
    logits = self.rory_lm_head(h)

    #if logits.shape[1] == 0:
      # special case for empty prompt
      #logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
    logits = [logits[0][-1]]

    if temperature < 1e-6:
      ret = logits.argmax(-1)
    else:
      logits = np.array(logits) / temperature
      logits[0] = np.exp(logits[0] - np.max(logits[0]))
      logits[0] = logits[0] / logits[0].sum()
      logits = logits.cumsum(1)
      logits = logits / logits[0][-1]
      logits = [logits]
      #can't get around not using tg here for e2e test?
      #maybe store the output in a file
      unif_samples = Tensor.rand(1, np.shape(logits)[0], 1)
      ##
      unif_samples = unif_samples.numpy()
      b = np.empty_like(logits,dtype=bool)
      for i in range(len(logits[0][0])):
        if unif_samples[0][0][0] >= logits[0][0][i]:
          b[0][0][i] = True
        else:
          b[0][0][i] = False

      b = b.sum(2)[0]
      ret = b
    return ret #why the realize? what calls this? the h hi loop?

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
    weights['rory_ln_f.weight'] = weights['ln_f.weight']
    weights['rory_ln_f.bias'] = weights['ln_f.bias']
    weights['rory_wte.weight'] = weights['wte.weight']
    weights['rory_wpe.weight'] = weights['wpe.weight']
    for i in range(12):
      weights['h.'+str(i)+'.rory_ln_1.weight'] = weights['h.'+str(i)+'.ln_1.weight']
      weights['h.'+str(i)+'.rory_ln_1.bias'] = weights['h.'+str(i)+'.ln_1.bias']
      weights['h.'+str(i)+'.rory_ln_2.weight'] = weights['h.'+str(i)+'.ln_2.weight']
      weights['h.'+str(i)+'.rory_ln_2.bias'] = weights['h.'+str(i)+'.ln_2.bias']
      weights['h.'+str(i)+'.rory_mlp.c_fc.weight'] = weights['h.'+str(i)+'.mlp.c_fc.weight']
      weights['h.'+str(i)+'.rory_mlp.c_fc.bias'] = weights['h.'+str(i)+'.mlp.c_fc.bias']
      weights['h.'+str(i)+'.rory_mlp.c_proj.weight'] = weights['h.'+str(i)+'.mlp.c_proj.weight']
      weights['h.'+str(i)+'.rory_mlp.c_proj.bias'] = weights['h.'+str(i)+'.mlp.c_proj.bias']
      weights['h.'+str(i)+'.rory_attn.c_attn.weight'] = weights['h.'+str(i)+'.attn.c_attn.weight']
      weights['h.'+str(i)+'.rory_attn.c_attn.bias'] = weights['h.'+str(i)+'.attn.c_attn.bias']
      weights['h.'+str(i)+'.rory_attn.c_proj.weight'] = weights['h.'+str(i)+'.attn.c_proj.weight']
      weights['h.'+str(i)+'.rory_attn.c_proj.bias'] = weights['h.'+str(i)+'.attn.c_proj.bias']
    model.rory_lm_head.weight = model.lm_head.weight #todo properly later
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
      tok = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), temperature).tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    return [rory_decode(x) for x in toks]

# **** main code ****
  
if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")
  default_prompt = "What is the answer to life, the universe, and everything?"
  #default_prompt = "What happened in 1939?"
  #should output:
  #It was a very fateful day.
  #When the Nazis occupied Poland in 1939....

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
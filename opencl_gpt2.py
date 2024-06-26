#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Union, Tuple
from tqdm import trange
import numpy as np
import math
import os
import pickle
import opencl_kernels as openclk
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch
opencl = True
med = False
dim = 768
if med == True:
  import opencl_kernels_med as openclk
  dim = 1024

MAX_CONTEXT = 128

tokens = open('tokens.txt','r',encoding="utf-8").readlines()
token_dict = dict()
max_token_length = -1
for i in range(len(tokens)): 
  s = tokens[i].replace("\n","").replace("/n","\n")
  token_dict[s] = i
  if len(s) > max_token_length:
    max_token_length = len(s)
def decode(index):
  ret = ""
  for i in index:
    ret+=tokens[i].replace("\n","").replace("/n","\n") #hack with linebreak
  return ret

def scaled_dot_product_attention(x, key, value, attn_mask=None,
                                  dropout_p:float=0.0, is_causal:bool=False):
  key = np.transpose(key,(0,1,3,2))
  x = np.float32(x)
  key = np.float32(key)
  value = np.float32(value)
  #qk = np.matmul(x,key)[0] # kernel below
  qk = openclk.matmul_t_3d(np.copy(x[0]),np.copy(key[0]))
  qk = qk / math.sqrt(np.shape(x)[-1])
  for x in range(len(qk)):
    for y in range(len(qk[0])):
      for z in range(len(qk[0][0])):
        if z > y: qk[x][y][z] -= np.inf
      qk[x][y] = np.exp(qk[x][y] - np.max(qk[x][y]))
      qk[x][y] = qk[x][y] / qk[x][y].sum()
  #qk = np.matmul(qk,value)
  qk = np.array([openclk.matmul_t_3d(np.copy(qk),np.copy(value[0]))])
  return qk

class Linear():
  def __init__(self, in_features, out_features, bias=True):
    # TODO: is this init good? torch inits to uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    self.bias = None
    self.weight = None

  def __call__(self,x):
    x = np.float32(x)
    self.weight = np.float32(self.weight)
    if self.bias is not None:
      self.bias = np.float32(self.bias)
    else:
      self.bias = np.zeros(np.shape(self.weight[1])).astype(np.float32)
    x = x[0]
    if np.shape(x)[0] == 1:
      ret = openclk.matvec2(x,self.weight,self.bias)
      if len(np.shape(ret)) == 1:
        ret = [ret] #todo
    else:
      #ret = np.matmul(x,self.weight) kernel below
      ret = openclk.matmul_t(x,self.weight)
      ret += self.bias
    ret = [ret]
    return ret
  
class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.bias = None
    self.key = key
    self.weight = None

  def __call__(self, x):
    x = np.float32(x)
    self.weight = np.float32(self.weight)
    self.bias = np.float32(self.bias)
    if np.shape(x)[1] == 1:
      x = x[0][0]
      #print("print rory x shape here =",np.shape(x))
      #mm = x - x.mean() #kernel below
      mm = openclk.minus_mean_multi(np.copy(x))
      #mm2 = np.float32(np.sqrt(np.mean(np.copy(mm)**2) + self.h[0].ln_1.eps)) #kernel below
      mm2 = openclk.sq_mean_sqrt(np.copy(mm))

      #x = ((mm * self.h[0].ln_1.weight) / mm2) + self.h[0].ln_1.bias #kernel below
      x = openclk.divide(np.copy(mm), mm2, self.weight, self.bias)
      x = [[x]]
      return x
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    #x = x.layernorm(eps=self.eps, axis=self.axis)
    #if not self.elementwise_affine: return x
    #return x * self.weight + self.bias
    #it doesnt work still with actual copy?
    for i in range(len(x[0])):
      #x[0][i] = (x[0][i] - np.mean(x[0][i])) / np.sqrt(np.mean((x[0][i] - np.mean(x[0][i]))**2) + self.eps)\
      #* self.weight + self.bias

      # todo all in one kernel instead of loop
      #mm = x[0][i] - np.mean(x[0][i]) #kernel below
      #mm = x[0][i] - np.mean(x[0][i]) # this causes an error, not sure why
      mm = openclk.minus_mean_multi(np.copy(x[0][i]))
      #mm2 = np.float32(np.sqrt(np.mean(np.copy(mm)**2) + self.eps)) #kernel below
      mm2 = openclk.sq_mean_sqrt(np.copy(mm))

      #x = ((mm * self.weight) / mm2) + self.bias #kernel below
      x[0][i] = openclk.divide(np.copy(mm), mm2, self.weight, self.bias)  
    return x

def encode(x):
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
    self.c_attn = Linear(dim, 3*dim, bias=True) #float32
    self.c_proj = Linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos, mask):
    #rory c_attn
    x = np.float32(x)
    self.c_attn.weight = np.float32(self.c_attn.weight)
    self.c_attn.bias = np.float32(self.c_attn.bias)

    if start_pos > 0:
      if np.shape(self.c_attn.weight) == (dim,dim*3):
        self.c_attn.weight = self.c_attn.weight.reshape(dim*3,dim) #have to do this for opencl...took way too long to realize
      xqkv = openclk.madd(x[0],self.c_attn.weight,self.c_attn.bias).reshape(dim*3) #todo make own kernel...
      xq = xqkv[0:self.dim]
      xk = xqkv[self.dim:2*self.dim]
      xk = xk.reshape(self.n_heads,self.head_dim)
      xv = xqkv[self.dim*2:]
      xv = xv.reshape(self.n_heads,self.head_dim)
      bsz, seqlen = 1,1
      # create kv cache
      if not hasattr(self, "cache_kv"):
        self.cache_kv = np.zeros(shape=[2, 1, MAX_CONTEXT, self.n_heads, self.head_dim])

      keys = self.cache_kv[0]
      values = self.cache_kv[1]
      
      keys[-1][start_pos] = xk
      values[-1][start_pos] = xv

      xq = xq.reshape(self.n_heads,1,self.head_dim)
      xk = xk.reshape(1,self.n_heads,self.head_dim)
      xv = xv.reshape(1,self.n_heads,self.head_dim)

      keys = np.resize(keys,(start_pos,self.n_heads,self.head_dim))
      values = np.resize(values,(start_pos,self.n_heads,self.head_dim))

      keys = np.concatenate([keys,xk])
      values = np.concatenate([values,xv])
      keys = keys.transpose(1,2,0)
      values = values.transpose(1,0,2)
      keys = np.float32(keys)
      xq = openclk.matmul2(xq,keys,np.shape(keys)[2])
      #for a in range(len(xq)):
      #  xq[a] = exp(xq[a] - np.max(xq[a]))
      #  xq[a] = xq[a] / xq[a].sum()
      #kernel below
      xq = openclk.minus_max(xq,(start_pos+1))

      values = values.astype(np.float32)
      #xq = np.matmul(xq,values) #kernel below
      xq = openclk.matmul3(xq,values,(start_pos+1))

      xq = xq.reshape((1,1,self.dim))
      self.c_proj.weight = np.float32(self.c_proj.weight)
      self.c_proj.bias = np.float32(self.c_proj.bias)

      #ret = np.matmul(xq,self.c_proj.weight) + self.c_proj.bias kernel below
      ret = openclk.matvec(xq,self.c_proj.weight,self.c_proj.bias)
      return ret
    
    else:
      #xqkv = np.matmul(x,self.c_attn.weight) #kernel below
      xqkv = [openclk.matmul_t(x[0],self.c_attn.weight)]
      xqkv += self.c_attn.bias
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

    keys = xk
    values = xv
    s = list(np.shape(keys))
    s[1] = MAX_CONTEXT
    new_cache = np.zeros(shape=s)
    new_cache = [np.copy(new_cache),np.copy(new_cache)]
    for i in range(len(keys[0])):
      new_cache[0][0][i] = keys[0][i]
      new_cache[1][0][i] = values[0][i]       
    self.cache_kv = new_cache
    xq = xq.transpose((0,2,1,3)) # same as (1,2) in tinygrad
    #can't numpy them outside this if!
    keys, values = keys.transpose((0,2,1,3)), values.transpose((0,2,1,3))
    
    xq = scaled_dot_product_attention(xq,keys,values,mask)
    xq = xq.transpose((0,2,1,3))
    #xq = xq.transpose(1, 2)
    xq = xq.reshape(bsz, seqlen, self.dim) #todo !
    ret = self.c_proj(xq)
    return ret
  
class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x):
    x = self.c_fc(x)
    for i in range(np.shape(x)[1]):
      # gelu() activation
      x[0][i] = 0.5 * x[0][i] * (1 + np.tanh(x[0][i] * 0.7978845608 * (1 + 0.044715 * x[0][i] * x[0][i])))
    ret = self.c_proj(x)
    ret = np.float64(ret) #todo, shouldnt need f64
    return ret
  
class Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

  def __call__(self, idx):
    ret = np.resize(self.weight,new_shape=(len(idx[0]),dim))
    return ret
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

  def __call__(self, idx):
    ret = np.empty((len(idx[0]),self.embed_size))
    for i in range(len(ret)):
      for j in range(len(ret[0])):
        ret[i][j] = self.weight[idx[0][i]][j]
    ret = [ret]
    return ret


class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim,n_heads) # done
    self.mlp = FeedForward(dim, 4*dim) #
    self.ln_1 = LayerNorm(dim,norm_eps)
    self.ln_2 = LayerNorm(dim,norm_eps)

  def __call__(self, x, start_pos, mask):
    h = np.copy(x)
    ln1 = self.ln_1(x)
    attn = self.attn(ln1,start_pos,mask)
    h += attn
    h2 = np.copy(h)
    ln2 = self.ln_2(h2) 
    mlp = self.mlp(ln2)
    ret = mlp + h
    return ret
    
class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding_2(vocab_size,dim)
    self.wpe = Embedding(max_seq_len,dim)

  def forward(self, tokens, start_pos, temperature:float=0.0,v_in=False):
    if not hasattr(self, 'allpos'): 
      self.allpos = np.arange(0, MAX_CONTEXT).reshape(1,-1)

    seqlen = tokens.shape[1]
    if seqlen > 1:
      mask = np.triu(np.full([seqlen,seqlen],1)) 
      mask = (mask - np.eye(seqlen)) * -math.inf
      mask[np.isnan(mask)] = 0
      np.where(np.isnan(mask), 0, mask) # inf * 0 = nan
      mask = [[mask]]
    else:
      mask = None
    
    tok_emb = self.wte(tokens) #rorys todo

    if start_pos > 0 and opencl:
      self.wpe.weight = np.float32(self.wpe.weight)
      self.wte.weight = np.float32(self.wte.weight)
      self.h[0].ln_1.weight = np.float32(self.h[0].ln_1.weight)
      self.h[0].ln_1.bias = np.float32(self.h[0].ln_1.bias)
      h = openclk.add(self.wte.weight,self.wpe.weight,start_pos,tokens[0][0])
      #h = self.h[0](h,start_pos,mask)
      #ln1 = self.h[0].ln_1(h)
      mm = openclk.minus_mean_multi(h)
      mm2 = openclk.sq_mean_sqrt(np.copy(mm))
      #x = ((mm * self.h[0].ln_1.weight) / mm2) + self.h[0].ln_1.bias #kernel below
      x = openclk.divide(np.copy(mm), mm2, self.h[0].ln_1.weight, self.h[0].ln_1.bias)
      x = [[x]]
      attn = self.h[0].attn(x,start_pos,mask)
      h = h.reshape(1,1,dim)
      h += attn
      h2 = np.copy(h)
      ln2 = self.h[0].ln_2(h2) 
      mlp = self.h[0].mlp(ln2)
      h = mlp + h  

      for i in range(1,len(self.h)):
        h = self.h[i](h, start_pos, mask)
      h = self.ln_f(h)
      logits = self.lm_head(h)
      logits = np.float64(logits) #todo shouldnt need f64
    else:
      tok_emb = self.wte(tokens) #rorys todo
      s = list(np.shape(self.allpos))
      s[1] = seqlen
      allpos_s = np.empty(s,dtype=np.int32)
      for i in range(seqlen):
        allpos_s[0][i] = self.allpos[0][start_pos + i]
      pos_emb = self.wpe(allpos_s)
      h = tok_emb + pos_emb


    #rory - h self.h is the 12 transformer blocks, so this is just forward through all
      for hi in self.h:
        h = hi(h, start_pos, mask)
      h = self.ln_f(h)
      logits = self.lm_head(h)
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
      #unif_samples = Tensor.rand(1, np.shape(logits)[0], 1)
      unif_samples = np.random.rand(1, np.shape(logits)[0], 1)
      #unif_samples = unif_samples.numpy()
      b = np.empty_like(logits,dtype=bool)
      for i in range(len(logits[0][0])):
        if unif_samples[0][0][0] >= logits[0][0][i]: #Tensor random gets [[[0.14280224]]] with 420 seed,
          b[0][0][i] = True
        else:
          b[0][0][i] = False
      b = b.sum(2)[0]
      ret = b
    return ret #why the realize? what calls this? the h hi loop?

  def __call__(self, tokens, start_pos, temperature:float=0.0,v_in=False):
    return self.forward(tokens, start_pos, temperature)

VOCAB_SIZE = 50257
class GPT2:
  @staticmethod
  def build():
    model = Transformer(n_layers=12,n_heads=12,dim=768,norm_eps=1e-5,vocab_size=VOCAB_SIZE) #small
    return GPT2(model)

  def __init__(self, model):
    self.model = model

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    prompt_tokens = encode(prompt)
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      if batch_size == 1 and len(toks[0][start_pos:]) == 1:
        tokens = np.array([[toks[0][start_pos]]])
      else:
        tokens = np.array(toks)
      tok = self.model(tokens, start_pos, temperature).tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    ret = [decode(x) for x in toks]
    return ret

# **** main code ****

if __name__ == "__main__":

  if os.path.exists("gpt2weights") == False:
    os.mkdir("gpt2weights")

  default_prompt = "What is the answer to life, the universe, and everything?"
  #default_prompt = "What happened in 1939?"
  # should output:
  # .... The Jewish people rejected

  #(tg random) should output:
  #It was a very fateful day.
  #When the Nazis occupied Poland in 1939....

  #Tensor.manual_seed(420) #don't need
  np.random.seed(28)

  #filehandler = open("weights_128.pickle", 'rb')  
  #gpt2 = pickle.load(filehandler)
  #filehandler = open("weights_med.pickle", 'rb')  
  #gpt2_med = pickle.load(filehandler)

  
  #gpt2_med = GPT2(model=Transformer(dim=1024,n_heads=16,n_layers=24,norm_eps=1e-5,vocab_size=50257))
  #gpt2 = GPT2(model=Transformer(dim=768,n_heads=12,n_layers=12,norm_eps=1e-5,vocab_size=50257))
  #gpt2 = GPT2.build()

  '''
  ##COPY WEIGHTS TO VAR AGAIN?
  #weights = torch_load(fetch(f'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'))
  weights = torch_load(fetch(f'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'))
  weights_med = torch_load(fetch(f'https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin'))
  for w in weights:
    print(w)

  gpt2_med.model.wte.weight = weights_med["wte.weight"].numpy()
  gpt2_med.model.wpe.weight = weights_med["wpe.weight"].numpy()
  gpt2_med.model.ln_f.weight = weights_med["ln_f.weight"].numpy()
  gpt2_med.model.ln_f.bias = weights_med["ln_f.bias"].numpy()
  gpt2_med.model.lm_head.weight = weights_med["wte.weight"].numpy().transpose(1,0)
  for i in range(24):
    gpt2_med.model.h[i].ln_1.weight = weights_med["h."+str(i)+".ln_1.weight"].numpy()
    gpt2_med.model.h[i].ln_1.bias = weights_med["h."+str(i)+".ln_1.bias"].numpy()
    gpt2_med.model.h[i].attn.bias = weights_med["h."+str(i)+".attn.bias"].numpy()
    gpt2_med.model.h[i].attn.c_attn.weight = weights_med["h."+str(i)+".attn.c_attn.weight"].numpy()
    gpt2_med.model.h[i].attn.c_attn.bias = weights_med["h."+str(i)+".attn.c_attn.bias"].numpy()
    gpt2_med.model.h[i].attn.c_proj.weight = weights_med["h."+str(i)+".attn.c_proj.weight"].numpy()
    gpt2_med.model.h[i].attn.c_proj.bias = weights_med["h."+str(i)+".attn.c_proj.bias"].numpy()
    gpt2_med.model.h[i].ln_2.weight = weights_med["h."+str(i)+".ln_2.weight"].numpy()
    gpt2_med.model.h[i].ln_2.bias = weights_med["h."+str(i)+".ln_2.bias"].numpy()
    gpt2_med.model.h[i].mlp.c_fc.weight = weights_med["h."+str(i)+".mlp.c_fc.weight"].numpy()
    gpt2_med.model.h[i].mlp.c_fc.bias = weights_med["h."+str(i)+".mlp.c_fc.bias"].numpy()
    #todo, why is this needed to keep gpt2 output equal?
    gpt2_med.model.h[i].mlp.c_proj.weight = np.zeros(np.shape(weights_med["h."+str(i)+".mlp.c_proj.weight"].numpy())).astype(np.float64)
    gpt2_med.model.h[i].mlp.c_proj.weight[:] = weights_med["h."+str(i)+".mlp.c_proj.weight"].numpy()[:]
    gpt2_med.model.h[i].mlp.c_proj.bias = np.zeros(np.shape(weights_med["h."+str(i)+".mlp.c_proj.bias"].numpy())).astype(np.float64)
    gpt2_med.model.h[i].mlp.c_proj.bias[:] = weights_med["h."+str(i)+".mlp.c_proj.bias"].numpy()[:]

  gpt2.model.wte.weight = weights["wte.weight"].numpy()
  gpt2.model.wpe.weight = weights["wpe.weight"].numpy()
  gpt2.model.ln_f.weight = weights["ln_f.weight"].numpy()
  gpt2.model.ln_f.bias = weights["ln_f.bias"].numpy()
  gpt2.model.lm_head.weight = weights["wte.weight"].numpy().transpose(1,0)
  for i in range(12):
    gpt2.model.h[i].ln_1.weight = weights["h."+str(i)+".ln_1.weight"].numpy()
    gpt2.model.h[i].ln_1.bias = weights["h."+str(i)+".ln_1.bias"].numpy()
    gpt2.model.h[i].attn.bias = weights["h."+str(i)+".attn.bias"].numpy()
    gpt2.model.h[i].attn.c_attn.weight = weights["h."+str(i)+".attn.c_attn.weight"].numpy()
    gpt2.model.h[i].attn.c_attn.bias = weights["h."+str(i)+".attn.c_attn.bias"].numpy()
    gpt2.model.h[i].attn.c_proj.weight = weights["h."+str(i)+".attn.c_proj.weight"].numpy()
    gpt2.model.h[i].attn.c_proj.bias = weights["h."+str(i)+".attn.c_proj.bias"].numpy()
    gpt2.model.h[i].ln_2.weight = weights["h."+str(i)+".ln_2.weight"].numpy()
    gpt2.model.h[i].ln_2.bias = weights["h."+str(i)+".ln_2.bias"].numpy()
    gpt2.model.h[i].mlp.c_fc.weight = weights["h."+str(i)+".mlp.c_fc.weight"].numpy()
    gpt2.model.h[i].mlp.c_fc.bias = weights["h."+str(i)+".mlp.c_fc.bias"].numpy()
    #todo, why is this needed to keep gpt2 output equal?
    gpt2.model.h[i].mlp.c_proj.weight = np.zeros(np.shape(weights["h."+str(i)+".mlp.c_proj.weight"].numpy())).astype(np.float64)
    gpt2.model.h[i].mlp.c_proj.weight[:] = weights["h."+str(i)+".mlp.c_proj.weight"].numpy()[:]
    gpt2.model.h[i].mlp.c_proj.bias = np.zeros(np.shape(weights["h."+str(i)+".mlp.c_proj.bias"].numpy())).astype(np.float64)
    gpt2.model.h[i].mlp.c_proj.bias[:] = weights["h."+str(i)+".mlp.c_proj.bias"].numpy()[:]


  with open('weights_128.pickle', 'wb') as outp:
    pickle.dump(gpt2, outp)
  with open('weights_med.pickle', 'wb') as outp:
    pickle.dump(gpt2_med, outp)
  ####END RORY
  '''
  #filehandler = open("weights_128.pickle", 'rb')  
  #gpt2 = pickle.load(filehandler)
  #filehandler = open("weights_med.pickle", 'rb')  
  #gpt2_med = pickle.load(filehandler)

  if med:
    filehandler = open("weights_med.pickle", 'rb')  
    gpt2_med = pickle.load(filehandler)
    texts = gpt2_med.generate(prompt=default_prompt, max_length=100, temperature=0.8, timing=None, batch_size=1)
    print('Generating text...')
    for i,text in enumerate(texts): print((f"Response {i}:", "green"), text)
    assert texts == [("What is the answer to life, the universe, and everything? There is no answer to life, the universe, and everything.\n\n"\
    "Enter the free thought experiment. The universe is composed entirely of subatomic particles that orbit the sun (or gas cloud or something) at the speeds of 3.7 kilometers per second. These particles interact with each other, creating light from ultraviolet light and creating electrical charges from the muon particles in the sun. These charges also create heat, light, and radiation from which electricity can be produced. The energy of these interactions varies")]

  if med == False:
    filehandler = open("weights_128.pickle", 'rb')  
    gpt2 = pickle.load(filehandler)
    texts = gpt2.generate(prompt=default_prompt, max_length=100, temperature=0.8, timing=None, batch_size=1)
    for i,text in enumerate(texts): print((f"Response {i}:", "green"), text)
    assert texts == [("What is the answer to life, the universe, and everything? "
    "But what is the answer to the mystery of Enlightenment? Does the only "
    "solution lie in a series of calls to agency? Do virtues and proper duties "
    "need to separate? How does a patient become his or her own individual conscience?\n\n"
    "What does the Universal Law mean? Why do some people do good and others contemptible? " 
    "How does the Universal Law maximize the efficiency of the health system? How does the "
    "Universal Law facilitate all of human virtue? What does it mean to be a man or a woman")]

  exit()
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
import math
rorys = True #todo args
import os

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

def rory_scaled_dot_product_attention(x, key, value, attn_mask=None,
                                  dropout_p:float=0.0, is_causal:bool=False):
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
  def __init__(self, in_features, out_features, bias=True,key="0"):
    if key != "0":
      print("rory linear init",key)
    # TODO: is this init good? torch inits to uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    self.weight = Tensor.zeros(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None
    self.w = None
    self.b = None
    self.key = key

  def __call__(self,x):
    #rory this is terrible atm obv
    w = self.weight.numpy()
    if self.bias:
      b = self.bias.numpy()
    if self.key != "0":
      if os.path.exists("gpt2weights/"+self.key+".txt") == False:
        print("writing file",self.key)
        f = open("gpt2weights/"+self.key+".txt", "w")
        f.write(str(self.weight.shape[0])+","+str(self.weight.shape[1])+"\n")
        for z in range(self.weight.shape[0]):
          for y in range(self.weight.shape[1]):
            f.write(str(w[z][y])+"\n")
        f.close()
    else:
      self.w = self.weight.numpy()
    
    if self.bias:
      if self.key != "0":
        if os.path.exists("gpt2weights/"+self.key+"_bias.txt") == False:
          print("writing bias file",self.key,"bias shape =",self.bias.shape)
          f = open("gpt2weights/"+self.key+"_bias.txt", "w")
          f.write(str(self.bias.shape)[1:-1]+"\n")
          if len(self.bias.shape) == 2:
            for z in range(self.bias.shape[0]):
              for y in range(self.bias.shape[1]):
                f.write(str(b[z][y])+"\n")
          if len(self.bias.shape) == 1:
            for z in range(self.bias.shape[0]):
              f.write(str(b[z])+"\n")
            f.close()
      else:
        self.b = self.bias.numpy()

    if self.b is None and self.key != "0" and os.path.exists("gpt2weights/"+self.key+"_bias.txt"):
      self.b = np.zeros(self.bias.shape)
      f = open("gpt2weights/"+self.key+"_bias.txt", 'r')
      print("loading bias for linear",self.key)
      lines = f.readlines()[1:]
      if len(self.bias.shape) == 2:
        for z in range(np.shape(b)[0]):
          for y in range(np.shape(b)[1]):
            self.b[z][y] = float(lines[z*np.shape(b)[1] + y].replace("\n",""))
      if len(self.bias.shape) == 1:
        for z in range(np.shape(b)[0]):
          self.b[z] = float(lines[z].replace("\n",""))
      f.close()
      

    if self.key != "0" and self.w is None:
      self.w = np.zeros(self.weight.shape) 
      f = open("gpt2weights/"+self.key+".txt", 'r')
      print("loading weights for linear",self.key)
      lines = f.readlines()[1:]
      print("rory shape is",np.shape(w))
      for z in range(np.shape(w)[0]):
        for y in range(np.shape(w)[1]):
          self.w[z][y] = float(lines[z*np.shape(w)[1] + y].replace("\n",""))
      f.close()
    w = np.copy(self.w)
    w = w.transpose()
    x = x[0]
    ret = np.matmul(x,w)
    if self.bias:
      for x in range(ret.shape[0]):
        ret[x] += self.b
    ret = [ret]
    return ret
  
class Rory_LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True,key="0"):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.bias = None
    self.key = key
    self.weight = None

  def __call__(self, x):
    if self.weight is None or self.bias is None:
      self.bias = np.zeros(self.normalized_shape)
      if os.path.exists("gpt2weights/layernorm"+str(self.key)+"_bias.txt") == False:        
        print("bias file missing")
        exit()
  
      f = open("gpt2weights/layernorm"+str(self.key)+"_bias.txt", 'r')
      lines = f.readlines()[1:]
      for i in range(len(lines)):
        self.bias[i] = lines[i]
    
    if self.weight is None:
      self.weight = np.zeros(self.normalized_shape)
      if os.path.exists("gpt2weights/layernorm"+str(self.key)+".txt") == False:        
        print("weights file missing")
        exit()
      
      f = open("gpt2weights/layernorm"+str(self.key)+".txt", 'r')
      lines = f.readlines()[1:]
      for i in range(len(lines)):
        self.weight[i] = lines[i]
  
    if np.shape(x)[1] == 1:
      x = x[0][0]
      x = (x - x.mean()) / np.sqrt(np.mean((x - x.mean())**2) + self.eps)\
      * self.weight + self.bias
      x = [[x]]
      return x
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    #x = x.layernorm(eps=self.eps, axis=self.axis)
    #if not self.elementwise_affine: return x
    #return x * self.weight + self.bias
    #it doesnt work still with actual copy?
    for i in range(len(x[0])):
      x[0][i] = (x[0][i] - np.mean(x[0][i])) / np.sqrt(np.mean((x[0][i] - np.mean(x[0][i]))**2) + self.eps)\
      * self.weight + self.bias
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

  def __call__(self, x, mask):
    return None

class Rory_Attention:
  def __init__(self, dim, n_heads,key="0"):
    self.key = key
    self.c_attn = Rory_Linear(dim, 3*dim, bias=True,key="at_0_"+self.key)
    self.c_proj = Rory_Linear(dim, dim, bias=True,key="at_1_"+self.key)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos:Variable, mask):
    if type(start_pos) is Variable:
      start_pos = start_pos.unbind()[1]
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

    if start_pos > 0:
      keys = self.cache_kv[0]
      values = self.cache_kv[1]
      ret = [keys,values]
      #terrible loop
      for a in range(len(ret)):
          for b in range(len(ret[0])):
              for c in range(start_pos+1,len(ret[0][0])):
                  ret[a][b][c] = np.zeros_like(ret[a][b][c])
          if start_pos > -1 and start_pos < len(ret[0][0]):
              ret[0][b][start_pos] = xk[0][0]
              ret[1][b][start_pos] = xv[0][0]
      new_cache = ret
      self.cache_kv = new_cache
      #new_cache = None #todo not needed?
      
      xq = xq.transpose((0,2,1,3)) # same as (1,2) in tinygrad

      s = list(np.shape(keys))
      s[1] = start_pos
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
  def __init__(self, dim, hidden_dim,key="0"):
    print("rory feedforward init key =",key)
    self.key = key
    self.c_fc = Rory_Linear(dim, hidden_dim, bias=True,key="ff_0_"+self.key)
    self.c_proj = Rory_Linear(hidden_dim, dim, bias=True,key="ff_1_"+self.key)

  def __call__(self, x):
    x = self.c_fc(x)
    for i in range(np.shape(x)[1]):
      # gelu() activation
      x[0][i] = 0.5 * x[0][i] * (1 + np.tanh(x[0][i] * 0.7978845608 * (1 + 0.044715 * x[0][i] * x[0][i])))
    ret = self.c_proj(x)
    return ret
  
class Rory_Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    print("rory emedding init")
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = None

  def __call__(self, idx):
    if not hasattr(self, 'vocab_counter'):
      self.vocab_counter = [[np.arange(start=0,stop=self.vocab_size)]]
    batch_size, seqlen = idx.shape
    if seqlen == 0:
      print("rory seq len is 0")
      exit()

    if self.weight is None:
      if os.path.exists("gpt2weights/embedding.txt") == False:
        print("weights not found")
        exit()
      #rory todo, move to init?
      self.weight = np.zeros([1024,768])
      f = open("gpt2weights/embedding.txt", 'r')
      lines = f.readlines()[1:]
      for y in range(1024):
        for x in range(768):
          self.weight[y][x] = lines[y*768 + x].replace("\n","")

    if idx.shape[1] == 1:
      b = np.repeat(False,self.vocab_size)
      b[idx] = True
      ret = [[np.matmul(b,self.weight)]]
      return ret
    
    b = np.empty((1,idx.shape[1],self.vocab_size),dtype=bool)
    b.fill(False)
    for i in range(len(b[0])):
      b[0][i][i] = True
    ret = np.matmul(b,self.weight)
    return ret
  
class Rory_Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = None

  def __call__(self, idx):
    if self.weight is None:
      #rory todo load in init!
      if os.path.exists("gpt2weights/embedding2.txt"):
        self.weight = np.empty([50257,768])
        print("txt exists loading embedding2")
        # load it
        f = open("gpt2weights/embedding2.txt", 'r')
        lines = f.readlines()[1:]
        for y in range(50257):
          for x in range(768):
            self.weight[y][x] = lines[768*y + x].replace("\n","")
      else:
        print("weights missing")
        exit()

    if not hasattr(self, 'vocab_counter'):
      self.vocab_counter = np.arange(self.vocab_size)
      self.vocab_counter = self.vocab_counter.reshape(1,1,self.vocab_size)
    idx_np = []
    for i in range(len(idx[0])):
      idx_np.append([idx[0][i]])
    idx_np = ([idx_np] == self.vocab_counter)
    ret = np.matmul(idx_np,self.weight)
    return ret


class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps,key="0"):
    self.attn = Attention(dim, n_heads)
    self.rory_attn = Rory_Attention(dim,n_heads,key=key)
    self.mlp = FeedForward(dim, 4*dim)
    self.rory_mlp = Rory_FeedForward(dim, 4*dim,key=key)
    self.ln_1 = LayerNorm(dim, norm_eps) #partly done
    self.rory_ln_1 = Rory_LayerNorm(dim,norm_eps,key="0_"+key)
    self.ln_2 = LayerNorm(dim, norm_eps) #done
    self.rory_ln_2 = Rory_LayerNorm(dim,norm_eps,key="1_"+key)

  def __call__(self, x, start_pos:Variable, mask):
    h = np.copy(x)
    ln1 = self.rory_ln_1(x)
    attn = self.rory_attn(ln1,start_pos,mask)
    h += attn
    h2 = np.copy(h)
    ln2 = self.rory_ln_2(h2) 
    mlp = self.rory_mlp(ln2)
    ret = mlp + h
    return ret
    
class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding(vocab_size, dim)
    self.rory_wte = Rory_Embedding_2(vocab_size,dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.rory_wpe = Rory_Embedding(max_seq_len,dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps,key=str(i)) for i in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.rory_ln_f = Rory_LayerNorm(dim,norm_eps,key="3")
    self.lm_head = Linear(dim, vocab_size, bias=False)
    self.rory_lm_head = Rory_Linear(dim, vocab_size, bias=False) #fix late
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens, start_pos:Variable, temperature:float=0.0,v_in=False):
    if not hasattr(self, 'allpos'): 
      self.allpos = np.arange(0, MAX_CONTEXT).reshape(1,-1)

    seqlen = tokens.shape[1]
    tok_emb = self.rory_wte(tokens) #rorys todo

    s = list(np.shape(self.allpos))
    s[1] = seqlen
    allpos_s = np.empty(s,dtype=np.int32)
    for i in range(seqlen):
      allpos_s[0][i] = self.allpos[0][start_pos.unbind()[1] + i]
    pos_emb = self.rory_wpe(allpos_s)
    h = tok_emb + pos_emb

    if seqlen > 1:
      mask = np.triu(np.full([seqlen,seqlen],1)) 
      mask = (mask - np.eye(seqlen)) * -math.inf
      mask[np.isnan(mask)] = 0
      np.where(np.isnan(mask), 0, mask) # inf * 0 = nan
      mask = [[mask]]
    else:
      mask = None

    #rory - h self.h is the 12 transformer blocks, so this is just forward through all
    for hi in self.h:
      h = hi(h, start_pos, mask)
    h = self.rory_ln_f(h)
    logits = self.rory_lm_head(h)

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

  def __call__(self, tokens, start_pos:Variable, temperature:float=0.0,v_in=False):
    return self.forward(tokens, start_pos, temperature)

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
        tokens = np.array([[toks[0][start_pos]]])
      else:
        tokens = np.array(toks)
      tok = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), temperature).tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    ret = [rory_decode(x) for x in toks]
    return ret

# **** main code ****

if __name__ == "__main__":

  if os.path.exists("gpt2weights") == False:
    os.mkdir("gpt2weights")

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
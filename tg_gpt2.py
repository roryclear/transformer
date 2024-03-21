#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Union, Tuple
from tqdm import trange
import numpy as np
import math
import os

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

class Linear():
  def __init__(self, in_features, out_features, bias=True,key="0"):
    # TODO: is this init good? torch inits to uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    self.key = key
    self.bias = None
    if os.path.exists("gpt2weights/"+self.key+"_bias.txt"):
      self.bias = np.zeros(out_features)
      f = open("gpt2weights/"+self.key+"_bias.txt", 'r')
      print("loading bias for linear",self.key)
      lines = f.readlines()[1:]
      for z in range(np.shape(self.bias)[0]):
        self.bias[z] = float(lines[z].replace("\n",""))
      f.close()

    if os.path.exists("gpt2weights/"+self.key+".txt"):
      self.weight = np.zeros([out_features,in_features]) 
      f = open("gpt2weights/"+self.key+".txt", 'r')
      print("loading weights for linear",self.key)
      lines = f.readlines()[1:]
      for z in range(np.shape(self.weight)[0]):
        for y in range(np.shape(self.weight)[1]):
          self.weight[z][y] = float(lines[z*np.shape(self.weight)[1] + y].replace("\n",""))
      f.close()
      self.weight = self.weight.transpose()

  def __call__(self,x):
    #rory this is terrible atm obv    
    x = x[0]
    ret = np.matmul(x,self.weight)
    if self.bias is not None:
      for x in range(ret.shape[0]):
        ret[x] += self.bias
    ret = [ret]
    return ret
  
class LayerNorm:
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
    self = self

  def __call__(self, x, mask):
    return None

class Attention:
  def __init__(self, dim, n_heads,key="0"):
    self.key = key
    self.c_attn = Linear(dim, 3*dim, bias=True,key="at_0_"+self.key)
    self.c_proj = Linear(dim, dim, bias=True,key="at_1_"+self.key)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos, mask):
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
    
    xq = scaled_dot_product_attention(xq,keys,values,mask)
    xq = xq.transpose((0,2,1,3))
    #xq = xq.transpose(1, 2)
    xq = xq.reshape(bsz, seqlen, self.dim) #todo !
    ret = self.c_proj(xq)
    return ret
  
class FeedForward:
  def __init__(self, dim, hidden_dim,key="0"):
    print("rory feedforward init key =",key)
    self.key = key
    self.c_fc = Linear(dim, hidden_dim, bias=True,key="ff_0_"+self.key)
    self.c_proj = Linear(hidden_dim, dim, bias=True,key="ff_1_"+self.key)

  def __call__(self, x):
    x = self.c_fc(x)
    for i in range(np.shape(x)[1]):
      # gelu() activation
      x[0][i] = 0.5 * x[0][i] * (1 + np.tanh(x[0][i] * 0.7978845608 * (1 + 0.044715 * x[0][i] * x[0][i])))
    ret = self.c_proj(x)
    return ret
  
class Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = None
    self.weight = np.zeros([self.vocab_size,self.embed_size])
    f = open("gpt2weights/embedding.txt", 'r')
    lines = f.readlines()[1:]
    for y in range(self.vocab_size):
      for x in range(self.embed_size):
        self.weight[y][x] = lines[y*self.embed_size + x].replace("\n","")

  def __call__(self, idx):
    if not hasattr(self, 'vocab_counter'):
      self.vocab_counter = [[np.arange(start=0,stop=self.vocab_size)]]
    batch_size, seqlen = idx.shape
    if seqlen == 0:
      print("rory seq len is 0")
      exit()

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
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = None
    self.weight = np.empty([self.vocab_size,self.embed_size])
    f = open("gpt2weights/embedding2.txt", 'r')
    lines = f.readlines()[1:]
    for y in range(self.vocab_size):
      for x in range(self.embed_size):
        self.weight[y][x] = lines[self.embed_size*y + x].replace("\n","")

  def __call__(self, idx):
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
    self.attn = Attention(dim,n_heads,key=key)
    self.mlp = FeedForward(dim, 4*dim,key=key)
    self.ln_1 = LayerNorm(dim,norm_eps,key="0_"+key)
    self.ln_2 = LayerNorm(dim,norm_eps,key="1_"+key)

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
    self.h = [TransformerBlock(dim, n_heads, norm_eps,key=str(i)) for i in range(n_layers)]
    self.ln_f = LayerNorm(dim,norm_eps,key="3")
    self.lm_head = Linear(dim, vocab_size, bias=False,key="transformer_linear")

  def forward(self, tokens, start_pos, temperature:float=0.0,v_in=False):
    if not hasattr(self, 'allpos'): 
      self.allpos = np.arange(0, MAX_CONTEXT).reshape(1,-1)

    seqlen = tokens.shape[1]
    tok_emb = self.wte(tokens) #rorys todo

    s = list(np.shape(self.allpos))
    s[1] = seqlen
    allpos_s = np.empty(s,dtype=np.int32)
    for i in range(seqlen):
      allpos_s[0][i] = self.allpos[0][start_pos + i]
    pos_emb = self.wpe(allpos_s)
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

  gpt2 = GPT2.build()

  texts = gpt2.generate(prompt=default_prompt, max_length=100, temperature=0.8, timing=None, batch_size=1)
  print('Generating text...')
  for i,text in enumerate(texts): print((f"Response {i}:", "green"), text)
  assert texts == [("What is the answer to life, the universe, and everything? "
  "But what is the answer to the mystery of Enlightenment? Does the only "
  "solution lie in a series of calls to agency? Do virtues and proper duties "
  "need to separate? How does a patient become his or her own individual conscience?\n\n"
  "What does the Universal Law mean? Why do some people do good and others contemptible? " 
  "How does the Universal Law maximize the efficiency of the health system? How does the "
  "Universal Law facilitate all of human virtue? What does it mean to be a man or a woman")]

  #assert for tg seed 420 unif samples
  '''
  assert texts == [("What is the answer to life, the universe, and everything?"
  "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
  "be a bit different. For one, you might take a longer view of the universe. "
  "But for other people — including scientists, artists, and other in-your-face "
  "people — your answer might be far more like: Life doesn't exist at all.\n\n"
  "Imagine you are a young person who just graduated from middle school and has "
  "never really pursued a career in astrophysics. You're on an eight")]
  '''
  exit()
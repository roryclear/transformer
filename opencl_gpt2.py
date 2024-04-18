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
opencl = True

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

def scaled_dot_product_attention(x, key, value):
  key = np.transpose(key,(0,2,1))
  #qk = np.matmul(x,key)[0] # kernel below
  qk = openclk.matmul_t_3d(np.copy(x),np.copy(key))
  
  qk = qk / math.sqrt(np.shape(x)[-1])
  for x in range(len(qk)):
    for y in range(len(qk[0])):
      for z in range(len(qk[0][0])):
        if z > y: qk[x][y][z] -= np.inf
      qk[x][y] = np.exp(qk[x][y] - np.max(qk[x][y]))
      qk[x][y] = qk[x][y] / qk[x][y].sum()
  #qk = np.matmul(qk,value)
  qk = np.array([openclk.matmul_t_3d(np.copy(qk),np.copy(value))])
  return qk

class Linear():
  def __init__(self, in_features, out_features, bias=True):
    self.bias = None
    self.weight = None

  def __call__(self,x):
    x = np.array(x) #todo
    if self.bias is None:
      self.bias = np.zeros(np.shape(self.weight[1])).astype(np.float32)
    if np.shape(x)[0] == 1:
      ret = openclk.matvec2(x,self.weight,self.bias)
      if len(np.shape(ret)) == 1:
        ret = [ret] #todo
    else:
      #ret = np.matmul(x,self.weight) kernel below
      ret = openclk.matmul_t(x,self.weight)
      ret += self.bias
    return ret
  
class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.bias = None
    self.weight = None

  def __call__(self, x):
    if np.shape(x)[0] == 1:
      #mm = x - x.mean() #kernel below
      mm = openclk.minus_mean_multi(np.copy(x))
      #mm2 = np.float32(np.sqrt(np.mean(np.copy(mm)**2) + self.eps)) #kernel below
      mm2 = openclk.sq_mean_sqrt(np.copy(mm))

      #x = ((mm * self.weight) / mm2) + self.bias #kernel below
      x = openclk.divide(np.copy(mm), mm2, self.weight, self.bias)
    else:
      assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
      #x = x.layernorm(eps=self.eps, axis=self.axis)
      #if not self.elementwise_affine: return x
      #return x * self.weight + self.bias
      #it doesnt work still with actual copy?
      for i in range(len(x)):
        #x[0][i] = (x[0][i] - np.mean(x[0][i])) / np.sqrt(np.mean((x[0][i] - np.mean(x[0][i]))**2) + self.eps)\
        #* self.weight + self.bias

        # todo all in one kernel instead of loop
        #mm = x[0][i] - np.mean(x[0][i]) #kernel below
        #mm = x[0][i] - np.mean(x[0][i]) # this causes an error, not sure why
        mm = openclk.minus_mean_multi(np.copy(x[i]))
        #mm2 = np.float32(np.sqrt(np.mean(np.copy(mm)**2) + self.eps)) #kernel below
        mm2 = openclk.sq_mean_sqrt(np.copy(mm))

        #x = ((mm * self.weight) / mm2) + self.bias #kernel below
        x[i] = openclk.divide(np.copy(mm), mm2, self.weight, self.bias)
    return x #todo

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

  def __call__(self, x, start_pos):
    x = np.array(x)
    if start_pos > 0:
      if np.shape(self.c_attn.weight) == (768,2304):
        self.c_attn.weight = self.c_attn.weight.reshape(2304,768) #have to do this for opencl...took way too long to realize
      #xqkv = np.matmul(x,self.c_attn.weight.reshape(768,2304)) + self.c_attn.bias #kernel below...doesnt work with changes
      xqkv = openclk.madd(x,self.c_attn.weight,self.c_attn.bias).reshape(2304) #todo make own kernel...
      xq = xqkv[0:self.dim]
      xk = xqkv[self.dim:2*self.dim]
      xk = xk.reshape(self.n_heads,self.head_dim)
      xv = xqkv[self.dim*2:]
      xv = xv.reshape(self.n_heads,self.head_dim)
      # create kv cache

      if not hasattr(self, "cache_kv"):
        exit() #todo never called?
        self.cache_kv = np.zeros(shape=[2, MAX_CONTEXT, self.n_heads, self.head_dim]).astype(np.float32)
      self.cache_kv = np.reshape(self.cache_kv,newshape=[2, MAX_CONTEXT, self.n_heads, self.head_dim]) #todo, resave file?
      keys = self.cache_kv[0]
      values = self.cache_kv[1]
      keys[start_pos] = xk
      values[start_pos] = xv
      
      xq = xq.reshape(self.n_heads,1,self.head_dim)
      xk = xk.reshape(1,self.n_heads,self.head_dim)
      xv = xv.reshape(1,self.n_heads,self.head_dim)

      keys = np.resize(keys,(start_pos,self.n_heads,self.head_dim))
      values = np.resize(values,(start_pos,self.n_heads,self.head_dim))

      keys = np.concatenate([keys,xk])
      values = np.concatenate([values,xv])
      keys = keys.transpose(1,2,0)
      values = values.transpose(1,0,2)
      
      #xq = np.matmul(xq,keys) / math.sqrt(self.head_dim) kernel below is same as
      xq = openclk.matmul2(xq,keys,np.shape(keys)[2])

      #for a in range(len(xq)):
      #  xq[a] = exp(xq[a] - np.max(xq[a]))
      #  xq[a] = xq[a] / xq[a].sum()
      #kernel below
      xq = openclk.minus_max(xq,(start_pos+1))
      
      #xq = np.matmul(xq,values) #kernel below
      xq = openclk.matmul3(xq,values,(start_pos+1))

      #xq = xq.reshape((1,1,self.dim))
      #ret = np.matmul(xq,self.c_proj.weight) + self.c_proj.bias #kernel below
      ret = openclk.matvec(xq,self.c_proj.weight,self.c_proj.bias)
      return ret

    else:
      #xqkv = np.matmul(x,self.c_attn.weight) #kernel below
      xqkv = openclk.matmul_t(x,self.c_attn.weight)
      xqkv += self.c_attn.bias
      xq = np.zeros(shape=(np.shape(xqkv)[0],self.dim)).astype(np.float32)
      for i in range(xq.shape[0]):
        xq[i] = xqkv[i][0:self.dim]
      xq = xq.reshape(xq.shape[0],self.n_heads,self.head_dim)
      xk = np.zeros(shape=(np.shape(xqkv)[0],self.dim)).astype(np.float32)
      for i in range(xk.shape[0]):
        xk[i] = xqkv[i][self.dim:2*self.dim]
      xk = xk.reshape(1,xk.shape[0],self.n_heads,self.head_dim)
      xv = np.zeros(shape=(1,np.shape(xqkv)[0],self.dim)).astype(np.float32)
      for i in range(xv.shape[1]):
        xv[0][i] = xqkv[i][self.dim*2:3*self.dim]
      xv = xv.reshape(1,xv.shape[1],self.n_heads,self.head_dim)
      bsz = 1
      seqlen, _, _ = xq.shape
    

    keys = xk
    values = xv
    s = list(np.shape(keys))
    s[1] = MAX_CONTEXT
    new_cache = np.zeros(shape=s).astype(np.float32)
    new_cache = [np.copy(new_cache),np.copy(new_cache)]
    for i in range(len(keys[0])):
      new_cache[0][0][i] = keys[0][i]
      new_cache[1][0][i] = values[0][i]       
    self.cache_kv = new_cache
    xq = xq.transpose((1,0,2)) # same as (1,2) in tinygrad
    #can't numpy them outside this if!
    keys, values = keys.transpose((0,2,1,3)), values.transpose((0,2,1,3))
    xq = scaled_dot_product_attention(xq,keys[0],values[0]) #todo
    xq = xq.transpose((0,2,1,3)) #todo
    xq = xq.reshape(seqlen, self.dim)
    ret = self.c_proj(xq)
    return ret
    
class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x):
    x = self.c_fc(x) #todo
    for i in range(np.shape(x)[0]):
      # gelu() activation
      x[i] = 0.5 * x[i] * (1 + np.tanh(x[i] * 0.7978845608 * (1 + 0.044715 * x[i] * x[i])))
    ret = self.c_proj(x)
    return ret
  
class Embedding: #todo, not used but variables are
  def __init__(self, vocab_size:int, embed_size:int):
    self = self

  def __call__(self, idx):
    return None
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

  def __call__(self, idx):
    ret = np.empty((len(idx),self.embed_size)).astype(np.float32)
    for i in range(len(ret)):
      for j in range(len(ret[0])):
        ret[i][j] = self.weight[idx[i]][j]
    return ret

class Mock_tg_rand:
  def __init__(self):
    self.index = 0
    file1 = open('random_nums.txt', 'r')
    self.lines = file1.readlines()

  def rand(self):
    ret = np.float32(self.lines[self.index])
    self.index+=1
    return ret

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim,n_heads) # done
    self.mlp = FeedForward(dim, 4*dim) #
    self.ln_1 = LayerNorm(dim,norm_eps)
    self.ln_2 = LayerNorm(dim,norm_eps)

  def __call__(self, x, start_pos):
    h = np.copy(x)
    ln1 = self.ln_1(x[0]) #todo
    attn = self.attn(ln1,start_pos)
    h += attn
    h2 = np.copy(h)
    ln2 = self.ln_2(h2[0]) #todo
    mlp = self.mlp(ln2) #todo
    ret = [mlp] + h #todo
    return ret
    
class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding_2(vocab_size,dim)
    self.wpe = Embedding(max_seq_len,dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for i in range(n_layers)]
    self.ln_f = LayerNorm(dim,norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)

  def convert(self):
    print("CONVERT")
    #self.wte.weight = np.float32(self.wte.weight)
    print(type(self.wte.weight[0][0]))
    #self.wpe.weight = np.float32(self.wpe.weight)
    print(type(self.wpe.weight[0][0]))
    #self.lm_head.weight = np.float32(self.lm_head.weight)
    print(type(self.lm_head.weight[0][0]))
    #self.ln_f.weight = np.float32(self.ln_f.weight)
    print(type(self.ln_f.weight[0]))
    #self.ln_f.bias = np.float32(self.ln_f.bias)
    print(type(self.ln_f.bias[0]))
    print("transformer block")
    for hi in self.h:
      #hi.attn.c_attn.weight = np.float32(hi.attn.c_attn.weight)
      print(type(hi.attn.c_attn.weight[0][0]))
      #hi.attn.c_attn.bias = np.float32(hi.attn.c_attn.bias)
      print(type(hi.attn.c_attn.bias[0]))
      #hi.attn.c_proj.weight = np.float32(hi.attn.c_proj.weight)
      print(type(hi.attn.c_proj.weight[0][0]))
      #hi.attn.c_proj.bias = np.float32(hi.attn.c_proj.bias)
      print(type(hi.attn.c_proj.bias[0]))
      #hi.mlp.c_fc.weight = np.float32(hi.mlp.c_fc.weight)
      print(type(hi.mlp.c_fc.weight[0][0]))
      #hi.mlp.c_fc.bias = np.float32(hi.mlp.c_fc.bias)
      print(type(hi.mlp.c_fc.bias[0]))
      #hi.mlp.c_proj.weight = np.float32(hi.mlp.c_proj.weight)
      print(type(hi.mlp.c_proj.weight[0][0]))
      #hi.mlp.c_proj.bias = np.float32(hi.mlp.c_proj.bias)
      print(type(hi.mlp.c_proj.bias[0]))
      #hi.ln_1.weight = np.float32(hi.ln_1.weight)
      print(type(hi.ln_1.weight[0]))
      #hi.ln_1.bias = np.float32(hi.ln_1.bias)
      print(type(hi.ln_1.bias[0]))
      #hi.ln_2.weight = np.float32(hi.ln_2.weight)
      print(type(hi.ln_2.weight[0]))
      #hi.ln_2.bias = np.float32(hi.ln_2.bias)
      print(type(hi.ln_2.bias[0]))

  def forward(self, tokens, start_pos, temperature:float=0.0):
    if not hasattr(self, 'allpos'): 
      self.allpos = np.arange(0, MAX_CONTEXT).reshape(1,-1)

    seqlen = tokens.shape[1]

    if start_pos > 0 and opencl:
      h = openclk.add(self.wte.weight,self.wpe.weight,start_pos,tokens[0][0])
      #h = self.h[0](h,start_pos,mask)
      #ln1 = self.h[0].ln_1(h)
      #todo, why do things need to be np.copied???
      #mm = h - np.mean(h) #kernel below
      mm = openclk.minus_mean_multi(h)
      #mm2 = np.float32(np.sqrt(np.mean(np.copy(mm)**2) + self.h[0].ln_1.eps)) #kernel below
      mm2 = openclk.sq_mean_sqrt(np.copy(mm))

      #x = ((mm * self.h[0].ln_1.weight) / mm2) + self.h[0].ln_1.bias #kernel below
      x = openclk.divide(np.copy(mm), mm2, self.h[0].ln_1.weight, self.h[0].ln_1.bias)
      attn = self.h[0].attn([x],start_pos)
      h = h.reshape(1,1,768)
      h += attn
      h2 = np.copy(h)
      ln2 = self.h[0].ln_2(h2[0]) #todo
      mlp = self.h[0].mlp(ln2) #todo
      h = mlp + h  

      for i in range(1,len(self.h)):
        h = self.h[i](h, start_pos)
      h = self.ln_f(h[0]) #todo
      logits = self.lm_head(h)
      logits = [logits] #todo
    else:
      tok_emb = self.wte(tokens[0]) #rorys todo
      tok_emb = [tok_emb] #todo
      s = list(np.shape(self.allpos))
      s[1] = seqlen
      pos_emb = np.resize(self.wpe.weight,new_shape=(seqlen,768))
      h = tok_emb + pos_emb
      #rory - h self.h is the 12 transformer blocks, so this is just forward through all
      for hi in self.h:
        h = hi(h, start_pos)
      h = self.ln_f(h[0]) #todo
      logits = self.lm_head(h)
      logits = [logits] #todo

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
      if use_tg_rand:
        unif_samples = [[[tg_rand.rand()]]]
      else:
        unif_samples = np.random.rand(1, 1, 1).astype(np.float32)
      b = np.empty_like(logits,dtype=bool)
      for i in range(len(logits[0][0])):
        if unif_samples[0][0][0] >= logits[0][0][i]: #Tensor random gets [[[0.14280224]]] with 420 seed,
          b[0][0][i] = True
        else:
          b[0][0][i] = False

      b = b.sum(2)[0]
      ret = b
    return ret #why the realize? what calls this? the h hi loop?

  def __call__(self, tokens, start_pos, temperature:np.float32=0.0,v_in=False):
    return self.forward(tokens, start_pos, temperature)

def pt(x):
  for _ in range(len(np.shape(x))):
    x = x[0]
  return str(type(x))

VOCAB_SIZE = 50257
class GPT2:
  @staticmethod
  def build():
    model = Transformer(n_layers=12,n_heads=12,dim=768,norm_eps=1e-5,vocab_size=VOCAB_SIZE) #small
    return GPT2(model)

  def __init__(self, model):
    self.model = model

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    #self.model.convert()
    #with open('weights_2.pickle', 'wb') as handle:
    #  pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if tg_rand:
      excepted_tokens = [198, 198, 1532, 345, 547, 281, 48782,\
      893, 48187, 11, 393, 655, 257, 33013, 11, 534, 3280,\
      1244, 307, 257, 1643, 1180, 13, 1114, 530, 11, 345,\
      1244, 1011, 257, 2392, 1570, 286, 262, 6881, 13,\
      887, 329, 584, 661, 851, 1390, 5519, 11, 7912,\
      11, 290, 584, 287, 12, 14108, 12, 2550, 661, 851,\
      534, 3280, 1244, 307, 1290, 517, 588, 25, 5155, 1595,\
      470, 2152, 379, 477, 13, 198, 198, 25153, 345, 389, 257,\
      1862, 1048, 508, 655, 18303, 422, 3504, 1524, 290, 468,\
      1239, 1107, 19189, 257, 3451, 287, 48782, 23154, 13, 921, 821, 319, 281, 3624]

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
      for i,t in enumerate(tok):
        if tg_rand:
          np.testing.assert_equal(tok[0],excepted_tokens[start_pos-13])  
        toks[i].append(t)
    ret = [decode(x) for x in toks]
    return ret

# **** main code ****

if __name__ == "__main__":
  #bc tinygrad doesnt work in windows, and opencl doesnt work on WSL
  use_tg_rand = True #mocks tg random function by just reading from a file
  default_prompt = "What is the answer to life, the universe, and everything?"
  #default_prompt = "What happened in 1939?"
  # should output:
  # .... The Jewish people rejected

  #(tg random) should output:
  #It was a very fateful day.
  #When the Nazis occupied Poland in 1939....

  #Tensor.manual_seed(420) #don't need
  np.random.seed(28)
  #filehandler = open("weights.obj", 'rb') 
  #filehandler = open("weights.pickle", 'rb')
  filehandler = open("weights.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  if use_tg_rand:
    tg_rand = Mock_tg_rand()
  print(type(gpt2))


  texts = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1)
  print('Generating text...')
  for i,text in enumerate(texts): print((f"Response {i}:", "green"), text)
  #assert for tg seed 420 unif samples
  if tg_rand:
    assert texts == [("What is the answer to life, the universe, and everything?"
    "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
    "be a bit different. For one, you might take a longer view of the universe. "
    "But for other people — including scientists, artists, and other in-your-face "
    "people — your answer might be far more like: Life doesn't exist at all.\n\n"
    "Imagine you are a young person who just graduated from middle school and has "
    "never really pursued a career in astrophysics. You're on an eight")]
  else:
    # for np random
    assert texts == [("What is the answer to life, the universe, and everything? "
    "But what is the answer to the mystery of Enlightenment? Does the only "
    "solution lie in a series of calls to agency? Do virtues and proper duties "
    "need to separate? How does a patient become his or her own individual conscience?\n\n"
    "What does the Universal Law mean? Why do some people do good and others contemptible? " 
    "How does the Universal Law maximize the efficiency of the health system? How does the "
    "Universal Law facilitate all of human virtue? What does it mean to be a man or a woman")]

  exit()
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
  qk = openclk.matmul_t_3d(np.copy(x),np.copy(key))
  qk = qk / math.sqrt(np.shape(x)[-1])
  for x in range(len(qk)):
    for y in range(len(qk[0])):
      for z in range(len(qk[0][0])):
        if z > y: qk[x][y][z] -= np.inf
      qk[x][y] = np.exp(qk[x][y] - np.max(qk[x][y]))
      qk[x][y] = qk[x][y] / qk[x][y].sum()
  qk = np.array(openclk.matmul_t_3d(np.copy(qk),np.copy(value)))
  return qk

def scaled_dot_product_attention_b(x, key, value):
  qk = openclk.matvec4(np.copy(x),np.copy(key))
  qk = qk / math.sqrt(np.shape(x)[-1])
  qk = np.array(openclk.matmul_t(np.copy(qk),np.copy(value)))
  return qk

class Linear():
  def __init__(self, in_features, out_features, bias=True):
    self.bias = None
    self.weight = None

  def __call__(self,x):
    return None
  
class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.bias = None
    self.weight = None

  def __call__(self, x):
    for i in range(len(x)):
      x[i] = openclk.kernel_0(np.copy(x[i]),self.weight, self.bias)
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

  def __call__(self, x, start_pos,od_out=False):
    x = np.array(x)
    if start_pos > 0:
      xqkv = openclk.matmul_t_b(x,self.c_attn.weight) + self.c_attn.bias
      xq = xqkv[0:self.dim]
      xk = xqkv[self.dim:2*self.dim]
      xk = xk.reshape(self.n_heads,self.head_dim)
      xv = xqkv[self.dim*2:]
      xv = xv.reshape(self.n_heads,self.head_dim)
      keys = self.cache_kv[0]
      values = self.cache_kv[1]
      keys[start_pos] = xk
      values[start_pos] = xv  
      keys = np.resize(keys,(start_pos,self.n_heads,self.head_dim))
      values = np.resize(values,(start_pos,self.n_heads,self.head_dim))
      keys = np.concatenate([keys,[xk]]) #todo
      values = np.concatenate([values,[xv]]) #todo  
      keys = keys.transpose(1,2,0) #todo, can we not do this?
      xq = openclk.matmul2(xq,keys,np.shape(keys)[2])
      xq = openclk.minus_max(xq,(start_pos+1))
      values = values.transpose(1,0,2)
      xq = openclk.matmul3(xq,values,(start_pos+1))
      ret = openclk.matvec(xq,self.c_proj.weight,self.c_proj.bias)
      return ret

    if od_out:
      #xqkv = np.matmul(x,self.c_attn.weight) #kernel below
      xqkv = openclk.matmul_t(x,self.c_attn.weight)
      xqkv += self.c_attn.bias
      xq = xqkv[:,:self.dim]
      xk = xqkv[:,self.dim:2*self.dim]
      xv = xqkv[:,2*self.dim:]
      xq = xq.reshape(len(xq),self.n_heads,self.head_dim)
      xk = xk.reshape(len(xk),self.n_heads,self.head_dim)
      xv = xv.reshape(len(xv),self.n_heads,self.head_dim)
      keys = xk
      values = xv
      s = list(np.shape(keys))
      s[0] = MAX_CONTEXT
      new_cache = np.zeros(shape=s).astype(np.float32)
      new_cache = [np.copy(new_cache),np.copy(new_cache)]
      for i in range(len(keys)):
        new_cache[0][i] = keys[i]
        new_cache[1][i] = values[i]       
      self.cache_kv = new_cache
      xq = xq[-1] #todo
      keys = keys[-1] #todo
      values = values[-1] #todo
      xq = scaled_dot_product_attention_b(xq,keys,values)
      #ret = np.matmul(x,self.weight) kernel below
      ret = openclk.matmul_t_c(xq,self.c_proj.weight)
      ret += self.c_proj.bias
      return ret
    #xqkv = np.matmul(x,self.c_attn.weight) #kernel below
    xqkv = openclk.matmul_t(x,self.c_attn.weight)
    xqkv += self.c_attn.bias
    xq = xqkv[:,:self.dim]
    xk = xqkv[:,self.dim:2*self.dim]
    xv = xqkv[:,2*self.dim:]
    xq = xq.reshape(len(xq),self.n_heads,self.head_dim)
    xk = xk.reshape(len(xk),self.n_heads,self.head_dim)
    xv = xv.reshape(len(xv),self.n_heads,self.head_dim)
    seqlen = len(xq)
  
    keys = xk
    values = xv
    s = list(np.shape(keys))
    s[0] = MAX_CONTEXT
    new_cache = np.zeros(shape=s).astype(np.float32)
    new_cache = [np.copy(new_cache),np.copy(new_cache)]
    for i in range(len(keys)):
      new_cache[0][i] = keys[i]
      new_cache[1][i] = values[i]       
    self.cache_kv = new_cache
    xq, keys, values = xq.transpose((1,0,2)), keys.transpose((1,0,2)), values.transpose((1,0,2))
    xq = scaled_dot_product_attention(xq,keys,values)
    xq = xq.transpose((1,0,2))
    xq = xq.reshape(seqlen, self.dim)

    #ret = np.matmul(x,self.weight) kernel below
    ret = openclk.matmul_t(xq,self.c_proj.weight)
    ret += self.c_proj.bias
    return ret
    
class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x):
    ret = openclk.matmul_t(x,self.c_fc.weight)
    ret += self.c_fc.bias
    x = ret
    for i in range(len(x)):
      # gelu() activation
      x[i] = 0.5 * x[i] * (1 + np.tanh(x[i] * 0.7978845608 * (1 + 0.044715 * x[i] * x[i])))
    x = np.array(x) #todo
    ret = openclk.matmul_t(x,self.c_proj.weight)
    ret += self.c_proj.bias
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
    return None
    
class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding_2(vocab_size,dim)
    self.wpe = Embedding(max_seq_len,dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for i in range(n_layers)]
    self.ln_f = LayerNorm(dim,norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)
    self.dim = dim

  def forward(self, tokens, start_pos, temperature:float=0.0):
    seqlen = len(tokens)
    if start_pos > 0:
      h = openclk.add(self.wte.weight,self.wpe.weight,start_pos,tokens[0])
      for i in range(len(self.h)):
        #inlined attn
        xq,xk,xv = openclk.kernel_2(np.copy(h),self.h[i].ln_1.weight, self.h[i].ln_1.bias,self.h[i].attn.c_attn.weight,\
        np.copy(self.h[i].attn.c_attn.bias),self.h[i].attn.dim)
        xk = xk.reshape(self.h[i].attn.n_heads,self.h[i].attn.head_dim)
        xv = xv.reshape(self.h[i].attn.n_heads,self.h[i].attn.head_dim)
        keys = self.h[i].attn.cache_kv[0]
        values = self.h[i].attn.cache_kv[1]
        keys[start_pos] = xk
        values[start_pos] = xv
        keys = np.resize(keys,(start_pos,self.h[i].attn.n_heads,self.h[i].attn.head_dim))
        values = np.resize(values,(start_pos,self.h[i].attn.n_heads,self.h[i].attn.head_dim))
        keys = np.concatenate([keys,[xk]]) #todo
        values = np.concatenate([values,[xv]]) #todo  
        keys = keys.transpose(1,2,0) #todo, can we not do this?
        xq = openclk.matmul2(xq,keys,np.shape(keys)[2])
        xq = openclk.minus_max(xq,(start_pos+1))
        values = values.transpose(1,0,2)
        xq = openclk.matmul3(xq,values,(start_pos+1))
        attn = openclk.matvec_b(xq,self.h[i].attn.c_proj.weight,np.copy(self.h[i].attn.c_proj.bias))
        h += attn
        #inlined attn
        h = openclk.kernel_1(h,self.h[i].ln_2.weight, self.h[i].ln_2.bias,self.h[i].mlp.c_fc.weight,np.copy(self.h[i].mlp.c_fc.bias)\
        ,self.h[i].mlp.c_proj.weight,np.copy(self.h[i].mlp.c_proj.bias))
      
      h = openclk.kernel_0(np.copy(h),self.ln_f.weight, self.ln_f.bias)
      logits = openclk.matvec2(h,self.lm_head.weight,np.zeros(np.shape(self.lm_head.weight[1])).astype(np.float32))
    else:
      tok_emb = self.wte(tokens)
      pos_emb = np.resize(self.wpe.weight,new_shape=(seqlen,self.dim))
      x = tok_emb + pos_emb
      #rory - h self.h is the 12 transformer blocks, so this is just forward through all
      for i in range(len(self.h)-1):
        h = np.copy(x) #todo
        for j in range(len(x)): #todo, kernel instead of loop
          x[j] = openclk.kernel_0(np.copy(x[j]),self.h[i].ln_1.weight, self.h[i].ln_1.bias)
        attn = self.h[i].attn(x,start_pos)
        h += attn
        x = np.copy(h)
        for j in range(len(x)):
          x[j] = openclk.kernel_0(np.copy(x[j]),self.h[i].ln_2.weight, self.h[i].ln_2.bias)
        x = openclk.matmul_t(x,self.h[i].mlp.c_fc.weight)
        x += self.h[i].mlp.c_fc.bias
        for j in range(len(x)):
          x[j] = 0.5 * x[j] * (1 + np.tanh(x[j] * 0.7978845608 * (1 + 0.044715 * x[j] * x[j])))
        x = openclk.matmul_t(x,self.h[i].mlp.c_proj.weight)
        x += self.h[i].mlp.c_proj.bias
        x += h
        ############
      h = np.copy(x[-1]) #todo
      for j in range(len(x)): #todo, kernel instead of loop
        x[j] = openclk.kernel_0(np.copy(x[j]),self.h[-1].ln_1.weight, self.h[-1].ln_1.bias)
      attn = self.h[-1].attn(x,start_pos,od_out=True)    
      h += attn
      x = np.copy(h)
      x = openclk.kernel_0(np.copy(x),self.h[-1].ln_2.weight, self.h[-1].ln_2.bias)
      x = openclk.matmul_t_c(x,self.h[-1].mlp.c_fc.weight)
      x += self.h[-1].mlp.c_fc.bias
      x = 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
      x = openclk.matmul_t_c(x,self.h[-1].mlp.c_proj.weight)
      x += self.h[-1].mlp.c_proj.bias
      x += h
      x = openclk.kernel_0(np.copy(x),self.ln_f.weight, self.ln_f.bias)
      logits = openclk.matmul_t_c(x,self.lm_head.weight)
    if temperature < 1e-6:
      ret = logits.argmax(-1)
    else:
      logits = np.array(logits) / temperature
      logits = np.exp(logits - np.max(logits))
      logits = logits / logits.sum()
      logits = logits.cumsum(0)
      logits = logits / logits[-1]
      if use_tg_rand:
        unif_samples = tg_rand.rand()
      else:
        unif_samples = np.random.rand().astype(np.float32)
      b = np.empty_like(logits,dtype=bool)
      for i in range(len(logits)):
        if unif_samples >= logits[i]:
          b[i] = True
        else:
          b[i] = False
      b = b.sum()
      ret = np.array(b)
    return ret

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

    toks = encode(prompt)
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      if batch_size == 1 and len(toks[start_pos:]) == 1:
        tokens = np.array([toks[start_pos]])
      else:
        tokens = np.array(toks)
      tok = self.model(tokens, start_pos, temperature).tolist()
      start_pos = len(toks)
      if tg_rand:
        np.testing.assert_equal(tok,excepted_tokens[start_pos-13])  
      toks.append(tok)
    return decode(toks)

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
  #filehandler = open("weights_2.pickle", 'rb')
  filehandler = open("weights.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  if use_tg_rand:
    tg_rand = Mock_tg_rand()
  print(type(gpt2))


  text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1)
  print('Generating text...')
  print((f"Response:", "green"), text)
  #assert for tg seed 420 unif samples
  if tg_rand:
    assert text == ("What is the answer to life, the universe, and everything?"
    "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
    "be a bit different. For one, you might take a longer view of the universe. "
    "But for other people — including scientists, artists, and other in-your-face "
    "people — your answer might be far more like: Life doesn't exist at all.\n\n"
    "Imagine you are a young person who just graduated from middle school and has "
    "never really pursued a career in astrophysics. You're on an eight")
  else:
    # for np random
    assert text == ("What is the answer to life, the universe, and everything? "
    "But what is the answer to the mystery of Enlightenment? Does the only "
    "solution lie in a series of calls to agency? Do virtues and proper duties "
    "need to separate? How does a patient become his or her own individual conscience?\n\n"
    "What does the Universal Law mean? Why do some people do good and others contemptible? " 
    "How does the Universal Law maximize the efficiency of the health system? How does the "
    "Universal Law facilitate all of human virtue? What does it mean to be a man or a woman")

  exit()
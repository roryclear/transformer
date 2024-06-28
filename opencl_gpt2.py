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
n_layers = 12
if med == True:
  n_layers = 24
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

def scaled_dot_product_attention(x, key, value):
  key = np.transpose(key,(0,2,1))
  qk = openclk.matmul_t_3d(x,key)
  qk = qk / math.sqrt(np.shape(x)[-1])
  for x in range(len(qk)):
    for y in range(len(qk[0])):
      for z in range(len(qk[0][0])):
        if z > y: qk[x][y][z] -= np.inf
      qk[x][y] = np.exp(qk[x][y] - np.max(qk[x][y]))
      qk[x][y] = qk[x][y] / qk[x][y].sum()
  qk = np.array(openclk.matmul_t_3d(qk,value))
  return qk

def scaled_dot_product_attention_b(x, key, value):
  qk = openclk.matvec4(x,key)
  qk = qk / math.sqrt(np.shape(x)[-1])
  qk = np.array(openclk.matmul_t(qk,value))
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
    self.key = key
    self.weight = None

  def __call__(self, x):
    for i in range(len(x)):
      exit()
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
      exit()
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

      xq = openclk.kernel_3(xq,keys)
      values = values.transpose(1,0,2)
      xq = openclk.matmul3(xq,values,(start_pos+1))
      #xq = openclk.matmul3_b(xq,values,(start_pos+1))

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
    #x = self.c_fc(x) #above
    for i in range(len(x)):  
      # gelu() activation
      x[i] = 0.5 * x[i] * (1 + np.tanh(x[i] * 0.7978845608 * (1 + 0.044715 * x[i] * x[i])))
    x = np.array(x) #todo

    ret = openclk.matmul_t(x,self.c_proj.weight)
    ret += self.c_proj.bias
    return ret
  
class Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self = self

  def __call__(self, idx):
    return None
  
class Mock_tg_rand:
  def __init__(self):
    self.index = 0
    file1 = open('random_nums.txt', 'r')
    self.lines = file1.readlines()

  def rand(self):
    ret = np.float32(self.lines[self.index])
    self.index+=1
    return ret
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

  def __call__(self, idx):
    ret = np.empty((len(idx),self.embed_size)).astype(np.float32)
    for i in range(len(ret)):
      for j in range(len(ret[0])):
        ret[i][j] = self.weight[idx[i]][j]
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

  def convert(self):
    print("CONVERT")
    self.wte.weight = np.float32(self.wte.weight)
    print(type(self.wte.weight[0][0]))
    self.wpe.weight = np.float32(self.wpe.weight)
    print(type(self.wpe.weight[0][0]))
    self.lm_head.weight = np.float32(self.lm_head.weight)
    print(type(self.lm_head.weight[0][0]))
    self.ln_f.weight = np.float32(self.ln_f.weight)
    print(type(self.ln_f.weight[0]))
    self.ln_f.bias = np.float32(self.ln_f.bias)
    print(type(self.ln_f.bias[0]))
    print("transformer block")
    for hi in self.h:
      hi.attn.c_attn.weight = np.float32(hi.attn.c_attn.weight)
      print(type(hi.attn.c_attn.weight[0][0]))
      hi.attn.c_attn.bias = np.float32(hi.attn.c_attn.bias)
      print(type(hi.attn.c_attn.bias[0]))
      hi.attn.c_proj.weight = np.float32(hi.attn.c_proj.weight)
      print(type(hi.attn.c_proj.weight[0][0]))
      hi.attn.c_proj.bias = np.float32(hi.attn.c_proj.bias)
      print(type(hi.attn.c_proj.bias[0]))
      hi.mlp.c_fc.weight = np.float32(hi.mlp.c_fc.weight)
      print(type(hi.mlp.c_fc.weight[0][0]))
      hi.mlp.c_fc.bias = np.float32(hi.mlp.c_fc.bias)
      print(type(hi.mlp.c_fc.bias[0]))
      hi.mlp.c_proj.weight = np.float32(hi.mlp.c_proj.weight)
      print(type(hi.mlp.c_proj.weight[0][0]))
      hi.mlp.c_proj.bias = np.float32(hi.mlp.c_proj.bias)
      print(type(hi.mlp.c_proj.bias[0]))
      hi.ln_1.weight = np.float32(hi.ln_1.weight)
      print(type(hi.ln_1.weight[0]))
      hi.ln_1.bias = np.float32(hi.ln_1.bias)
      print(type(hi.ln_1.bias[0]))
      hi.ln_2.weight = np.float32(hi.ln_2.weight)
      print(type(hi.ln_2.weight[0]))
      hi.ln_2.bias = np.float32(hi.ln_2.bias)
      print(type(hi.ln_2.bias[0]))

  def forward(self, tokens, start_pos, temperature:float=0.0,v_in=False):
    if hasattr(self, 'ln_1_weights') == False:
      print("copying ln_1_weights")
      self.ln_1_weights = self.h[0].ln_1.weight
      for i in range(1,n_layers):
        self.ln_1_weights = np.concatenate((self.ln_1_weights,self.h[i].ln_1.weight))
    if hasattr(self, 'ln_1_bias') == False:
      print("copying ln_1_bias")
      self.ln_1_bias = self.h[0].ln_1.bias
      for i in range(1,n_layers):
        self.ln_1_bias = np.concatenate((self.ln_1_bias,self.h[i].ln_1.bias))
    if hasattr(self, 'attn_c_attn_bias') == False: #2304,
      print("copying attn_c_attn_bias")
      self.attn_c_attn_bias = self.h[0].attn.c_attn.bias
      for i in range(1,n_layers):
        self.attn_c_attn_bias = np.concatenate((self.attn_c_attn_bias,self.h[i].attn.c_attn.bias))
    if hasattr(self, 'attn_c_proj_bias') == False:
      print("copying attn_c_proj_bias")
      self.attn_c_proj_bias = self.h[0].attn.c_proj.bias
      for i in range(1,n_layers):
        self.attn_c_proj_bias = np.concatenate((self.attn_c_proj_bias,self.h[i].attn.c_proj.bias))
    if hasattr(self, 'ln_2_weight') == False: #768,
      print("copying ln_2_weight")
      self.ln_2_weight = self.h[0].ln_2.weight
      for i in range(1,n_layers):
        self.ln_2_weight = np.concatenate((self.ln_2_weight,self.h[i].ln_2.weight))
    if hasattr(self, 'ln_2_bias') == False: #768,
      print("copying ln_2_bias")
      self.ln_2_bias = self.h[0].ln_2.bias
      for i in range(1,n_layers):
        self.ln_2_bias = np.concatenate((self.ln_2_bias,self.h[i].ln_2.bias))
    if hasattr(self, 'mlp_c_fc_bias') == False: #768,
      print("copying mlp_c_fc_bias")
      self.mlp_c_fc_bias = self.h[0].mlp.c_fc.bias
      for i in range(1,n_layers):
        self.mlp_c_fc_bias = np.concatenate((self.mlp_c_fc_bias,self.h[i].mlp.c_fc.bias))
    if hasattr(self, 'attn_c_attn_weight') == False: #768*2304
      print("copying attn_c_attn_weight")
      self.attn_c_attn_weight = self.h[0].attn.c_attn.weight.transpose(1,0).flatten()
      for i in range(1,n_layers):
        self.attn_c_attn_weight = np.concatenate((self.attn_c_attn_weight,\
        self.h[i].attn.c_attn.weight.transpose(1,0).flatten()))
    if hasattr(self, 'attn_c_proj_weight') == False: #768*2304
      print("copying attn_c_proj_weight")
      self.attn_c_proj_weight = self.h[0].attn.c_proj.weight.flatten()
      for i in range(1,n_layers):
        self.attn_c_proj_weight = np.concatenate((self.attn_c_proj_weight,\
        self.h[i].attn.c_proj.weight.flatten()))
    if hasattr(self, 'mlp_c_fc_weight') == False: #768*2304
      print("copying mlp_c_fc_weight")
      self.mlp_c_fc_weight = self.h[0].mlp.c_fc.weight.transpose(1,0).flatten()
      for i in range(1,n_layers):
        self.mlp_c_fc_weight = np.concatenate((self.mlp_c_fc_weight,\
        self.h[i].mlp.c_fc.weight.transpose(1,0).flatten()))
    if hasattr(self, 'mlp_c_proj_weight') == False: #768*2304
      print("copying mlp.c_proj.weight")
      self.mlp_c_proj_weight = self.h[0].mlp.c_proj.weight.flatten()
      for i in range(1,n_layers):
        self.mlp_c_proj_weight = np.concatenate((self.mlp_c_proj_weight,\
        self.h[i].mlp.c_proj.weight.flatten()))
    if hasattr(self, 'mlp_c_proj_bias') == False: #768
      print("copying mlp_c_proj_bias")
      self.mlp_c_proj_bias = self.h[0].mlp.c_proj.bias
      for i in range(1,n_layers):
        self.mlp_c_proj_bias = np.concatenate((self.mlp_c_proj_bias,\
        self.h[i].mlp.c_proj.bias))
    # 2D !
    #if hasattr(self, 'attn_c_attn_weight') == False:
      #print("FFFFFFSSSS attn_c_attn_weight")
      #self.attn_c_attn_weight = np.concatenate((self.h[0].attn.c_attn.weight.flatten(),self.h[1].attn.c_attn.weight.flatten()))
    seqlen = len(tokens)
    if start_pos > 0:
      h = openclk.add(self.wte.weight,self.wpe.weight,start_pos,tokens[0])
      for i in range(len(self.h)):
        self.h[i].attn.c_proj.weight = self.h[i].attn.c_proj.weight.flatten()
        self.h[i].mlp.c_proj.weight = self.h[i].mlp.c_proj.weight.flatten()
        self.h[i].mlp.c_proj.bias = self.h[i].mlp.c_proj.bias.flatten()

        if hasattr(self, 'attn_cache_kv') == False: #128, 12, 64 -> 
          self.attn_cache_kv = np.concatenate((self.h[0].attn.cache_kv[0].flatten(),self.h[0].attn.cache_kv[1].flatten()))
          for i in range(1,n_layers):
            self.attn_cache_kv = np.concatenate((self.attn_cache_kv,self.h[i].attn.cache_kv[0].flatten()))
            self.attn_cache_kv = np.concatenate((self.attn_cache_kv,self.h[i].attn.cache_kv[1].flatten()))

      h = openclk.kernel_4(h,self.ln_1_weights,\
      self.ln_1_bias,\
      self.attn_c_attn_bias,self.h[0].attn.dim,\
      start_pos,\
      self.attn_c_proj_bias,\
      self.ln_2_weight, self.ln_2_bias,\
      self.mlp_c_fc_bias,\
      self.attn_c_attn_weight,\
      self.attn_cache_kv,
      self.attn_c_proj_weight,
      self.mlp_c_fc_weight,
      self.mlp_c_proj_weight,self.mlp_c_proj_bias)
        
      h = openclk.kernel_3(h,self.ln_f.weight, self.ln_f.bias)
      self.lm_head.weight = self.lm_head.weight.flatten()
      logits = openclk.matvec2(h,self.lm_head.weight)
    else:
      tok_emb = self.wte(tokens) #rorys todo
      pos_emb = np.resize(self.wpe.weight,new_shape=(seqlen,dim))
      x = tok_emb + pos_emb

      #rory - h self.h is the 12 transformer blocks, so this is just forward through all
      for i in range(len(self.h)-1):
        h = np.copy(x) #todo
        for j in range(len(x)): #todo, kernel instead of loop
          x[j] = openclk.kernel_0(x[j],self.h[i].ln_1.weight, self.h[i].ln_1.bias)
        attn = self.h[i].attn(x,start_pos)
        h += attn
        x = np.copy(h)

        for j in range(len(x)):
          x[j] = openclk.kernel_0(x[j],self.h[i].ln_2.weight, self.h[i].ln_2.bias)

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
        x[j] = openclk.kernel_0(x[j],self.h[-1].ln_1.weight, self.h[-1].ln_1.bias)
      attn = self.h[-1].attn(x,start_pos,od_out=True)   
      attn = attn[-1]
      h += attn
      x = np.copy(h)

      x = openclk.kernel_0(x,self.h[-1].ln_2.weight, self.h[-1].ln_2.bias)

      x = openclk.matmul_t_c(x,self.h[-1].mlp.c_fc.weight)

      x += self.h[-1].mlp.c_fc.bias

      x = 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
      x = openclk.matmul_t_c(x,self.h[-1].mlp.c_proj.weight)
      x += self.h[-1].mlp.c_proj.bias
      x += h

      x = openclk.kernel_0(x,self.ln_f.weight, self.ln_f.bias)

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
      #unif_samples = unif_samples.numpy()
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

VOCAB_SIZE = 50257
class GPT2:
  @staticmethod
  def build():
    model = Transformer(n_layers=12,n_heads=12,dim=768,norm_eps=1e-5,vocab_size=VOCAB_SIZE) #small
    return GPT2(model)

  def __init__(self, model):
    self.model = model

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    expected_tokens = [198, 198, 1532, 345, 547, 281, 48782,\
    893, 48187, 11, 393, 655, 257, 33013, 11, 534, 3280,\
    1244, 307, 257, 1643, 1180, 13, 1114, 530, 11, 345,\
    1244, 1011, 257, 2392, 1570, 286, 262, 6881, 13,\
    887, 329, 584, 661, 851, 1390, 5519, 11, 7912,\
    11, 290, 584, 287, 12, 14108, 12, 2550, 661, 851,\
    534, 3280, 1244, 307, 1290, 517, 588, 25, 5155, 1595,\
    470, 2152, 379, 477, 13, 198, 198, 25153, 345, 389, 257,\
    1862, 1048, 508, 655, 18303, 422, 3504, 1524, 290, 468,\
    1239, 1107, 19189, 257, 3451, 287, 48782, 23154, 13, 921, 821, 319, 281, 3624]

    expected_tokens_b = [198, 198,\
    1026, 373, 257, 845, 46873, 1110, 13, 198, 198, 2215, 262,
    19147, 12030, 12873, 287, 24414, 11, 262, 6771, 547, 407, 3142,\
    284, 670, 287, 262, 17590, 11, 645, 2300, 703, 881, 484, 2227,\
    284, 13, 383, 1917, 2627, 1598, 618, 262, 5103, 1664, 286, 262,\
    309, 9116, 4623, 268, 4618, 11, 543, 925, 281, 3113, 329, 262,\
    11908, 12, 1273, 14414, 41460, 11, 3414, 617, 19008, 284, 262,\
    24718, 25931, 13, 198, 198, 464, 2551, 373, 2077, 706, 257, 1327,\
    6531, 1022, 262, 7570, 4479, 338, 1964, 5531, 290, 12267, 7602, 11, 290, 373, 1912, 319, 262]

    expected_tokens_med = [198, 198, 1544, 468, 262, 2694,\
    290, 262, 481, 284, 3853, 475, 339, 2391, 2314, 2222,\
    2241, 284, 466, 340, 13, 679, 318, 7787, 284, 307,\
    3436, 290, 7787, 284, 2222, 1854, 656, 340, 13, 679,\
    318, 7787, 284, 307, 33046, 290, 7787, 284, 307, 8606,\
    13, 198, 198, 4864, 11, 339, 318, 407, 3436, 287, 465,\
    3252, 286, 5287, 13, 198, 198, 22210, 4952, 502, 326,\
    3252, 2125, 470, 262, 6808, 2728, 286, 262, 1917, 13,\
    198, 198, 2025, 37560, 198, 198, 4864, 11, 611, 356, 804,\
    9211, 356, 1064, 326, 4213, 836, 470, 423, 284, 307, 7042, 287]

    toks = encode(prompt)
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      if batch_size == 1 and len(toks[start_pos:]) == 1:
        tokens = np.array([toks[start_pos]])
      else:
        tokens = np.array(toks)
      tok = self.model(tokens, start_pos, temperature).tolist()
      start_pos = len(toks)
      if med == False:
        if default_prompt == "What is the answer to life, the universe, and everything?":
          np.testing.assert_equal(tok,expected_tokens[start_pos-13])
        else:
          np.testing.assert_equal(tok,expected_tokens_b[start_pos-5])
      else:
        np.testing.assert_equal(tok,expected_tokens_med[start_pos-13])  
      toks.append(tok)
    return decode(toks)

# **** main code ****

if __name__ == "__main__":
  use_tg_rand = True

  if use_tg_rand:
    tg_rand = Mock_tg_rand()

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
    gpt2_med.model.convert()

    for i in range(24):
      gpt2_med.model.h[i].attn.c_attn.weight = np.asfortranarray(gpt2_med.model.h[i].attn.c_attn.weight)
      gpt2_med.model.h[i].attn.c_proj.weight = np.asfortranarray(gpt2_med.model.h[i].attn.c_proj.weight)
      gpt2_med.model.h[i].mlp.c_fc.weight = np.asfortranarray(gpt2_med.model.h[i].mlp.c_fc.weight)
      gpt2_med.model.h[i].mlp.c_proj.weight = np.asfortranarray(gpt2_med.model.h[i].mlp.c_proj.weight)

    text = gpt2_med.generate(prompt=default_prompt, max_length=100, temperature=0.8, timing=None, batch_size=1)
    print('Generating text...')
    print((f"Response:", "green"), text)
    assert text == ("What is the answer to life, the universe, and everything?\n\n"
    "He has the ability and the will to choose but he simply cannot bring himself to"
    " do it. He is afraid to be alone and afraid to bring others into it. "
    "He is afraid to be misunderstood and afraid to be rejected.\n\n"
    "However, he is not alone in his fear of failure.\n\n"
    "Something tells me that fear isn't the root cause of the problem.\n\n"
    "An Idea\n\nHowever, if we look deeper we find that ideas don't have to be formed in")

  if med == False:
    filehandler = open("weights_128.pickle", 'rb')  
    gpt2 = pickle.load(filehandler)
    gpt2.model.convert()
    filehandler = open("weights.pickle", 'rb')  
    gpt2_og = pickle.load(filehandler)


    for i in range(12):
      gpt2.model.h[i].attn.c_attn.weight = np.asfortranarray(gpt2.model.h[i].attn.c_attn.weight)
      gpt2.model.h[i].attn.c_proj.weight = np.asfortranarray(gpt2.model.h[i].attn.c_proj.weight)
      gpt2.model.h[i].mlp.c_fc.weight = np.asfortranarray(gpt2.model.h[i].mlp.c_fc.weight)
      gpt2.model.h[i].mlp.c_proj.weight = np.asfortranarray(gpt2.model.h[i].mlp.c_proj.weight)

    text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1)
    print((f"Response:", "green"), text)
    assert text == ("What is the answer to life, the universe, and everything?"
    "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
    "be a bit different. For one, you might take a longer view of the universe. "
    "But for other people — including scientists, artists, and other in-your-face "
    "people — your answer might be far more like: Life doesn't exist at all.\n\n"
    "Imagine you are a young person who just graduated from middle school and has "
    "never really pursued a career in astrophysics. You're on an eight")

  exit()
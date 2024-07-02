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
import pyopencl as cl
opencl = True

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
mf = cl.mem_flags

med = False
dim = 768
n_heads = 12
if med == True:
  import opencl_kernels_med as openclk
  dim = 1024
  n_heads = 16

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

class Linear():
  def __init__(self, in_features, out_features, bias=True):
    self.bias = None
    self.weight = None

  
class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.bias = None
    self.key = key
    self.weight = None

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
      
class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

class Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self = self

  def __call__(self, idx):
    return None
  
class Mock_tg_rand:
  def __init__(self):
    self.seed = 420

  def rand(self):
    self.seed += 1
    rng = np.random.default_rng(self.seed)
    rng_np_buffer = rng.random(size=1, dtype=np.float32).astype(dtype=np.float32, copy=False)
    return rng_np_buffer[0]
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

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
    if hasattr(self, 'ln_1_weight') == False:
      print("copying ln_1_weight")
      self.ln_1_weight = []
      for i in range(len(self.h)):
        self.ln_1_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].ln_1.weight))

    if hasattr(self, 'ln_1_bias') == False:
      print("copying ln_1_bias")
      self.ln_1_bias = []
      for i in range(len(self.h)):
        self.ln_1_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].ln_1.bias))

    if hasattr(self, 'attn_c_attn_weight') == False:
      print("copying attn_c_attn_weight")
      self.attn_c_attn_weight = []
      for i in range(len(self.h)):
        self.attn_c_attn_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_attn.weight.transpose(1,0).flatten()))

    if hasattr(self, 'attn_c_attn_bias') == False: #dont use yet
      print("copying attn_c_attn_bias")
      self.attn_c_attn_bias = []
      for i in range(len(self.h)):
        self.attn_c_attn_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_attn.bias))
    
    if hasattr(self, 'attn_c_proj_weight') == False:
      print("copying attn_c_proj_weight")
      self.attn_c_proj_weight = []
      for i in range(len(self.h)):
        self.attn_c_proj_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_proj.weight.flatten()))
    
    if hasattr(self, 'attn_c_proj_bias') == False:
      print("copying attn_c_proj_bias")
      self.attn_c_proj_bias = []
      for i in range(len(self.h)):
        self.attn_c_proj_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_proj.bias))

    if hasattr(self, 'ln_2_weight') == False:
      print("copying ln_2_weight")
      self.ln_2_weight = []
      for i in range(len(self.h)):
        self.ln_2_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].ln_2.weight))

    if hasattr(self, 'ln_2_bias') == False:
      print("copying ln_2_bias")
      self.ln_2_bias = []
      for i in range(len(self.h)):
        self.ln_2_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].ln_2.bias))

    if hasattr(self, 'mlp_c_fc_bias') == False:
      print("copying mlp_c_fc_bias")
      self.mlp_c_fc_bias = []
      for i in range(len(self.h)):
        self.mlp_c_fc_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_fc.bias))

    if hasattr(self, 'attn_c_attn_weight') == False:
      print("copying attn_c_attn_weight")
      self.attn_c_attn_weight = []
      for i in range(len(self.h)):
        self.attn_c_attn_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_attn.weight.transpose(1,0).flatten()))

    if hasattr(self, 'mlp_c_proj_weight') == False:
      print("copying mlp_c_proj_weight")
      self.mlp_c_proj_weight = []
      for i in range(len(self.h)):
        self.mlp_c_proj_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_proj.weight.flatten()))

    if hasattr(self, 'mlp_c_proj_bias') == False:
      print("copying mlp_c_proj_bias")
      self.mlp_c_proj_bias = []
      for i in range(len(self.h)):
        self.mlp_c_proj_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_proj.bias))

    if hasattr(self, 'ln_f_weight') == False:
      print("copying ln_f_weight")
      self.ln_f_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ln_f.weight)
    
    if hasattr(self, 'ln_f_bias') == False:
      print("copying ln_f_bias")
      self.ln_f_bias = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ln_f.bias)

    if hasattr(self, 'mlp_c_fc_weight') == False:
      print("copying mlp_c_fc_weight")
      self.mlp_c_fc_weight = []
      for i in range(len(self.h)):
        self.mlp_c_fc_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_fc.weight.transpose(1,0).flatten()))

    if hasattr(self, 'lm_head_weight') == False:
      print("copying lm_head_weight")
      self.lm_head_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lm_head.weight.flatten())

    # 2D !
    #if hasattr(self, 'attn_c_attn_weight') == False:
      #print("FFFFFFSSSS attn_c_attn_weight")
      #self.attn_c_attn_weight = np.concatenate((self.h[0].attn.c_attn.weight.flatten(),self.h[1].attn.c_attn.weight.flatten()))
    seqlen = len(tokens)
    if start_pos > 0:
      if hasattr(self, 'attn_cache_kv') == False:
        print("copying attn_cache_kv")
        self.attn_cache_kv = []
        for i in range(len(self.h)):
          self.attn_cache_kv.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.concatenate((\
          self.h[i].attn.cache_kv[0].flatten(),self.h[i].attn.cache_kv[1].flatten()))))

      h = openclk.add(self.wte.weight,self.wpe.weight,start_pos,tokens[0])
      for i in range(len(self.h)):
        self.h[i].attn.c_proj.weight = self.h[i].attn.c_proj.weight.flatten()
        self.h[i].mlp.c_proj.weight = self.h[i].mlp.c_proj.weight.flatten()
        self.h[i].mlp.c_proj.bias = self.h[i].mlp.c_proj.bias.flatten()

      attn_dim = dim
      for i in range(0,len(self.h)):
        h = openclk.kernel_2(h,self.ln_1_weight[i],\
        self.ln_1_bias[i],self.attn_c_attn_weight[i],\
        self.attn_c_attn_bias[i],attn_dim,\
        self.attn_cache_kv[i],start_pos,\
        self.attn_c_proj_weight[i],self.attn_c_proj_bias[i],\
        self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],\
        self.mlp_c_proj_weight[i],self.mlp_c_proj_bias[i])
      h = openclk.kernel_3(h,self.ln_f_weight, self.ln_f_bias)
      if temperature < 1e-6:
        logits = openclk.matvec2(h,self.lm_head_weight) #todo
        ret = logits.argmax(-1)
      else:
        logits = openclk.matvec2(h,self.lm_head_weight,temperature)
        unif_samples = tg_rand.rand()
        ret = openclk.kernel_6(logits,unif_samples).astype(np.int32)[0]    
        return ret
    else:
      n_tokens = len(tokens)
      x = openclk.tok_emb(tokens,self.wte.weight,self.wpe.weight)

      #rory - h self.h is the 12 transformer blocks, so this is just forward through all
      for i in range(len(self.h)-1):
        h = np.copy(x) #todo
        x = openclk.kernel_0_b(x,self.h[i].ln_1.weight, self.h[i].ln_1.bias,n_tokens)
        xqkv = openclk.matmul_t_b(x,self.h[i].attn.c_attn.weight,n_tokens,self.attn_c_attn_bias[i])
        xq = xqkv[:,:dim]
        xk = xqkv[:,dim:2*dim]
        xv = xqkv[:,2*dim:]
        xq = xq.reshape(n_tokens,self.h[i].attn.n_heads,self.h[i].attn.head_dim)
        xk = xk.reshape(n_tokens,self.h[i].attn.n_heads,self.h[i].attn.head_dim)
        xv = xv.reshape(n_tokens,self.h[i].attn.n_heads,self.h[i].attn.head_dim)
        values = xv
        s = list(np.shape(xk))
        s[0] = MAX_CONTEXT
        new_cache = np.zeros(shape=s).astype(np.float32)
        new_cache = [np.copy(new_cache),np.copy(new_cache)]
        for j in range(len(xk)):
          new_cache[0][j] = xk[j]
          new_cache[1][j] = values[j]       
        self.h[i].attn.cache_kv = new_cache
        xq, values = xq.transpose((1,0,2)), values.transpose((1,0,2))
        xq = openclk.matmul_t_3d_c(xq,xk)
        xq = openclk.minus_sum_3d(xq,n_tokens)
        xq = openclk.matmul_t_3d(xq,values,n_tokens)
        xq = openclk.transpose(xq,n_tokens)
        h = openclk.matmul_t_e(xq,self.h[i].attn.c_proj.weight,self.attn_c_proj_bias[i],n_tokens,h)
        xq = None
        attn = None
        x = np.copy(h)

        x = openclk.kernel_0_b(x,self.h[i].ln_2.weight, self.h[i].ln_2.bias,n_tokens,True)
        x = openclk.matmul_t_d2(x,self.h[i].mlp.c_fc.weight,self.mlp_c_fc_bias[i])
        x = openclk.matmul_t_d(x,self.h[i].mlp.c_proj.weight,self.mlp_c_proj_bias[i],h)
        ############
      h = np.copy(x[-1]) #todo
      x = openclk.kernel_0_b(x,self.h[-1].ln_1.weight, self.h[-1].ln_1.bias,n_tokens,True)
      #attn = self.h[-1].attn(x)

      xqkv = openclk.matmul_t_f(x,self.attn_c_attn_weight[-1],n_tokens,self.attn_c_attn_bias[-1])
      xq = xqkv[:,:self.h[-1].attn.dim]
      xk = xqkv[:,self.h[-1].attn.dim:2*self.h[-1].attn.dim]
      xv = xqkv[:,2*self.h[-1].attn.dim:]
      xq = xq.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      xk = xk.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      xv = xv.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      new_cache = np.zeros(shape=s).astype(np.float32)
      new_cache = [np.copy(new_cache),np.copy(new_cache)]
      for i in range(len(xk)):
        new_cache[0][i] = xk[i]
        new_cache[1][i] = xv[i]         
      self.h[-1].attn.cache_kv = new_cache
      xq = xq[-1] #todo
      xk = xk[-1] #todo
      xv = xv[-1] #todo
      qk = openclk.matvec4(xq,xk)
      xq = openclk.matmul_t(qk,xv)
      x = openclk.matmul_t_c2(xq,self.h[-1].attn.c_proj.weight,self.attn_c_proj_bias[-1],h)
      x = openclk.kernel_0(x,self.ln_2_weight[-1], self.ln_2_bias[-1])

      x = openclk.matmul_t_c3(x,self.h[-1].mlp.c_fc.weight,self.mlp_c_fc_bias[-1])
      x = openclk.matmul_t_c2(x,self.h[-1].mlp.c_proj.weight,self.mlp_c_proj_bias[-1],h)
      h = None
      x = openclk.kernel_0(x,self.ln_f_weight, self.ln_f_bias)

    if temperature < 1e-6:
      logits = openclk.matmul_t_c(x,self.lm_head.weight) #todo
      ret = logits.argmax(-1)
    else:
      logits = openclk.matmul_t_c(x,self.lm_head.weight,temperature,True)
      unif_samples = tg_rand.rand()
      ret = openclk.kernel_6(logits,unif_samples).astype(np.int32)[0]
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
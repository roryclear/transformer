#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Union, Tuple
from tqdm import trange
import numpy as np
import math
import os
import pickle
import opencl_kernels# as openclk
import pyopencl as cl
opencl = True

openclk = opencl_kernels.Opencl_Kernels()

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
mf = cl.mem_flags

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
  
class Embedding: #todo, not used but variables are
  def __init__(self, vocab_size:int, embed_size:int):
    self = self

  def __call__(self, idx):
    return None
  
class Embedding_2: #todo crutch
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size, self.embed_size = vocab_size, embed_size

class Mock_tg_rand:
  def __init__(self):
    self.seed = 420

  def rand(self):
    self.seed += 1
    rng = np.random.default_rng(self.seed)
    rng_np_buffer = rng.random(size=1, dtype=np.float32).astype(dtype=np.float32, copy=False)
    return rng_np_buffer[0]

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
    #self.ln_1_weights = None

  def forward(self, tokens, start_pos, temperature:float=0.8,n_tokens=444):
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

    if hasattr(self, 'mlp_c_fc_weight') == False:
      print("copying mlp_c_fc_weight")
      self.mlp_c_fc_weight = []
      for i in range(len(self.h)):
        self.mlp_c_fc_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_fc.weight.transpose(1,0).flatten()))

    if hasattr(self, 'mlp_c_fc_bias') == False:
      print("copying mlp_c_fc_bias")
      self.mlp_c_fc_bias = []
      for i in range(len(self.h)):
        self.mlp_c_fc_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_fc.bias))

    if hasattr(self, 'mlp_c_proj_bias') == False:
      print("copying mlp_c_proj_bias")
      self.mlp_c_proj_bias = []
      for i in range(len(self.h)):
        self.mlp_c_proj_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_proj.bias))

    if hasattr(self, 'mlp_c_proj_weight') == False:
      print("copying mlp_c_proj_weight")
      self.mlp_c_proj_weight = []
      for i in range(len(self.h)):
        self.mlp_c_proj_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_proj.weight.flatten()))

    if hasattr(self, 'ln_f_weight') == False:
      print("copying ln_f_weight")
      self.ln_f_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ln_f.weight)

    if hasattr(self, 'ln_f_bias') == False:
      print("copying ln_f_bias")
      self.ln_f_bias = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.ln_f.bias)

    if hasattr(self, 'lm_head_weight') == False:
      print("copying lm_head_weight")
      self.lm_head_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lm_head.weight.flatten())
    
    if hasattr(self, 'wte_weight') == False:
      print("copying self_wte_weight")
      self.wte_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.wte.weight)

    if hasattr(self, 'wpe_weight') == False:
      print("copying self_wpe_weight")
      self.wpe_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.wpe.weight)

    if start_pos > 0:
      h = openclk.add(self.wte_weight,self.wpe_weight,start_pos,tokens[0])
      attn_dim = 768
      for i in range(0,len(self.h)):
        #inlined attn
        h = openclk.kernel_2(h,self.ln_1_weight[i],\
        self.ln_1_bias[i],self.attn_c_attn_weight[i],\
        self.attn_c_attn_bias[i],attn_dim,\
        self.h[i].attn.cache_kv,start_pos,\
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
      x = openclk.tok_emb(tokens,self.wte_weight,self.wpe.weight,n_tokens)
      for i in range(len(self.h)-1):
        h = x
        x = openclk.kernel_0_b(x,self.h[i].ln_1.weight, self.h[i].ln_1.bias,n_tokens)
        xqkv = openclk.matmul_t_f(x,self.attn_c_attn_weight[i],n_tokens,self.attn_c_attn_bias[i])
        new_cache = np.zeros((2*MAX_CONTEXT*12*64)).astype(np.float32)
        new_cache = openclk.copy_to_cache_b(xqkv,new_cache,n_tokens,MAX_CONTEXT)
        self.h[i].attn.cache_kv = new_cache
        xq = xqkv[:,:self.dim]
        xk = xqkv[:,self.dim:2*self.dim]
        xv = xqkv[:,2*self.dim:]

        xq = xq.flatten() #todo remove
        xv = xv.flatten()
        xq,xv = openclk.transpose_b(xq,n_tokens,xv)
        xq = openclk.matmul_t_3d_c(xq,xk,n_tokens)
        xq = openclk.minus_sum_3d(xq,n_tokens)
        xq = openclk.matmul_t_3d(xq,xv,n_tokens)
        xq = openclk.transpose(xq,n_tokens)
        h = openclk.matmul_t_e(xq,self.h[i].attn.c_proj.weight,self.attn_c_proj_bias[i],n_tokens,h)
        x = np.copy(h)
        x = openclk.kernel_0_b(x,self.h[i].ln_2.weight, self.h[i].ln_2.bias,n_tokens,True)
        x = openclk.matmul_t_d2(x,self.h[i].mlp.c_fc.weight,self.mlp_c_fc_bias[i],n_tokens)
        x = openclk.matmul_t_d(x,self.h[i].mlp.c_proj.weight,self.mlp_c_proj_bias[i],h,n_tokens)
        ############
      h = np.copy(x[-1]) 
      x = openclk.kernel_0_b(x,self.h[-1].ln_1.weight, self.h[-1].ln_1.bias,n_tokens,True)
      xqkv = openclk.matmul_t_f(x,self.attn_c_attn_weight[-1],n_tokens,self.attn_c_attn_bias[-1])
      xq = xqkv[:,:self.h[-1].attn.dim]
      xk = xqkv[:,self.h[-1].attn.dim:2*self.h[-1].attn.dim]
      xv = xqkv[:,2*self.h[-1].attn.dim:]
      xq = xq.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      xk = xk.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      xv = xv.reshape(n_tokens,self.h[-1].attn.n_heads,self.h[-1].attn.head_dim)
      new_cache = np.zeros((2*MAX_CONTEXT*12*64)).astype(np.float32)
      new_cache = openclk.copy_to_cache(xk,xv,new_cache,n_tokens,MAX_CONTEXT)
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

  def __call__(self, tokens, start_pos, temperature:np.float32=0.0,n_tokens=1):
    return self.forward(tokens, start_pos, temperature,n_tokens)

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

      excepted_tokens_b = [198, 198,\
      1026, 373, 257, 845, 46873, 1110, 13, 198, 198, 2215, 262,
      19147, 12030, 12873, 287, 24414, 11, 262, 6771, 547, 407, 3142,\
      284, 670, 287, 262, 17590, 11, 645, 2300, 703, 881, 484, 2227,\
      284, 13, 383, 1917, 2627, 1598, 618, 262, 5103, 1664, 286, 262,\
      309, 9116, 4623, 268, 4618, 11, 543, 925, 281, 3113, 329, 262,\
      11908, 12, 1273, 14414, 41460, 11, 3414, 617, 19008, 284, 262,\
      24718, 25931, 13, 198, 198, 464, 2551, 373, 2077, 706, 257, 1327,\
      6531, 1022, 262, 7570, 4479, 338, 1964, 5531, 290, 12267, 7602, 11, 290, 373, 1912, 319, 262]

    toks = encode(prompt)
    start_pos = 0
    n_tokens = len(toks)
    for _ in trange(max_length, disable=(timing==True)):
      if batch_size == 1 and len(toks[start_pos:]) == 1:
        tokens = np.array([toks[start_pos]])
      else:
        tokens = np.array(toks)
      tok = self.model(tokens, start_pos, temperature, n_tokens).tolist()
      start_pos = len(toks)
      
      if tg_rand:
        if default_prompt == "What is the answer to life, the universe, and everything?":
          np.testing.assert_equal(tok,excepted_tokens[start_pos-13])
        else:
          np.testing.assert_equal(tok,excepted_tokens_b[start_pos-5])
      
      toks.append(tok)
    return decode(toks)

# **** main code ****

if __name__ == "__main__":
  #bc tinygrad doesnt work in windows, and opencl doesnt work on WSL
  default_prompt = "What is the answer to life, the universe, and everything?"
  #default_prompt = "What happened in 1939?"

  #Tensor.manual_seed(420) #don't need
  np.random.seed(28)
  #filehandler = open("weights.obj", 'rb') 
  #filehandler = open("weights_2.pickle", 'rb')
  filehandler = open("weights.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  tg_rand = Mock_tg_rand()


  text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1)
  print('Generating text...')
  print((f"Response:", "green"), text)
  #assert for tg seed 420 unif samples
  assert text == ("What is the answer to life, the universe, and everything?"
  "\n\nIf you were an astrophysicist, or just a physicist, your answer might "
  "be a bit different. For one, you might take a longer view of the universe. "
  "But for other people — including scientists, artists, and other in-your-face "
  "people — your answer might be far more like: Life doesn't exist at all.\n\n"
  "Imagine you are a young person who just graduated from middle school and has "
  "never really pursued a career in astrophysics. You're on an eight")

  exit()
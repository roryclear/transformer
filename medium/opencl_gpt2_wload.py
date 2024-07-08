#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Union, Tuple
from tqdm import trange
import numpy as np
import math
import os
import pickle
import opencl_kernels
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch
import pyopencl as cl
opencl = True

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
mf = cl.mem_flags


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
  def __init__():
    return None

class LayerNorm:
  def __init__():
    return None

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
  def __init__():
    return None
      
class FeedForward:
  def __init__():
    return None

class Embedding:
  def __init__():
    return None

  def __call__():
    return None
  
class Rand:
  def __init__(self):
    self.seed = 420

  def rand(self):
    self.seed += 1
    rng = np.random.default_rng(self.seed)
    rng_np_buffer = rng.random(size=1, dtype=np.float32).astype(dtype=np.float32, copy=False)
    return rng_np_buffer[0]
  
class Embedding_2: #todo crutch
  def __init__():
    return None

class TransformerBlock:
  def __init__():
    return None
    
class Transformer:
  def __init__():
    return None

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

  def forward(self, tokens, start_pos, temperature:float=0.8,n_tokens=444):
    if hasattr(self, 'ln_1_weight') == False:
      print("copying ln_1_weight")
      self.ln_1_weight = []
      for i in range(len(self.h)):
        self.ln_1_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_1_weight[i]))

    if hasattr(self, 'ln_1_bias') == False:
      print("copying ln_1_bias")
      self.ln_1_bias = []
      for i in range(len(self.h)):
        self.ln_1_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_1_bias[i]))

    if hasattr(self, 'attn_c_attn_weight') == False:
      print("copying attn_c_attn_weight")
      self.attn_c_attn_weight = []
      for i in range(len(self.h)):
        self.attn_c_attn_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.attn_c_attn_weight[i].transpose(1,0).flatten()))

    if hasattr(self, 'attn_c_attn_bias') == False: #dont use yet
      print("copying attn_c_attn_bias")
      self.attn_c_attn_bias = []
      for i in range(len(self.h)):
        self.attn_c_attn_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.attn_c_attn_bias[i]))
    
    if hasattr(self, 'attn_c_proj_weight') == False:
      print("copying attn_c_proj_weight")
      self.attn_c_proj_weight = []
      self.attn_c_proj_weight2 = []
      for i in range(len(self.h)):
        self.attn_c_proj_weight2.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].attn.c_proj.weight))
        self.attn_c_proj_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.attn_c_proj_weight[i].flatten()))

    if hasattr(self, 'attn_c_proj_bias') == False:
      print("copying attn_c_proj_bias")
      self.attn_c_proj_bias = []
      for i in range(len(self.h)):
        self.attn_c_proj_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.attn_c_proj_bias[i]))

    if hasattr(self, 'ln_2_weight') == False:
      print("copying ln_2_weight")
      self.ln_2_weight = []
      for i in range(len(self.h)):
        self.ln_2_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_2_weight[i]))

    if hasattr(self, 'ln_2_bias') == False:
      print("copying ln_2_bias")
      self.ln_2_bias = []
      for i in range(len(self.h)):
        self.ln_2_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_2_bias[i]))

    if hasattr(self, 'mlp_c_fc_bias') == False:
      print("copying mlp_c_fc_bias")
      self.mlp_c_fc_bias = []
      for i in range(len(self.h)):
        self.mlp_c_fc_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.mlp_c_fc_bias[i]))

    if hasattr(self, 'attn_c_attn_weight') == False:
      print("copying attn_c_attn_weight")
      self.attn_c_attn_weight = []
      for i in range(len(self.h)):
        self.attn_c_attn_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.attn_c_attn_weight[i].transpose(1,0).flatten()))

    if hasattr(self, 'mlp_c_proj_weight_unf') == False: #todo, what difference does the .flatten() make to the kernel? transposed?
      print("copying mlp_c_proj_weight_unf")
      self.mlp_c_proj_weight_unf = []
      self.mlp_c_proj_weight = []
      for i in range(len(self.h)):
        self.mlp_c_proj_weight_unf.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[i].mlp.c_proj.weight))
        self.mlp_c_proj_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.mlp_c_proj_weight[i].flatten()))

    if hasattr(self, 'mlp_c_proj_bias') == False:
      print("copying mlp_c_proj_bias")
      self.mlp_c_proj_bias = []
      for i in range(len(self.h)):
        self.mlp_c_proj_bias.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.mlp_c_proj_bias[i]))

    if hasattr(self, 'ln_f_weight') == False:
      print("copying ln_f_weight")
      self.ln_f_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_f_weight)
    
    if hasattr(self, 'ln_f_bias') == False:
      print("copying ln_f_bias")
      self.ln_f_bias = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.ln_f_bias)

    if hasattr(self, 'mlp_c_fc_weight') == False:
      print("copying mlp_c_fc_weight")
      self.mlp_c_fc_weight = []
      for i in range(len(self.h)):
        self.mlp_c_fc_weight.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.mlp_c_fc_weight[i].transpose(1,0).flatten()))

    if hasattr(self, 'lm_head_weight') == False:
      print("copying lm_head_weight")
      self.lm_head_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.lm_head_weight.flatten())

    if hasattr(self, 'lm_head_weight_unf') == False: #todo
      print("copying lm_head_weight_unf")
      self.lm_head_weight_unf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.lm_head_weight)

    if hasattr(self, 'wte_weight') == False:
      print("copying self_wte_weight")
      self.wte_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.wte_weight)

    if hasattr(self, 'wpe_weight') == False:
      print("copying self_wpe_weight")
      self.wpe_weight = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpt2_2.model.wpe_weight)

    if hasattr(self, 'attn_cache_kv') == False:
      print("creating attn_cache_kv")
      self.attn_cache_kv = []
      for i in range(len(self.h)):
        a = np.zeros((2*MAX_CONTEXT*n_heads*64)).astype(np.float32)
        self.attn_cache_kv.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a))

    if start_pos > 0:
      h = openclk.add(self.wte_weight,self.wpe_weight,start_pos,tokens[0])
      attn_dim = dim
      for i in range(0,len(self.h)):
        h = openclk.kernel_0(h,self.ln_1_weight[i],\
        self.ln_1_bias[i],self.attn_c_attn_weight[i],\
        self.attn_c_attn_bias[i],attn_dim,\
        self.attn_cache_kv[i],start_pos,\
        self.attn_c_proj_weight[i],self.attn_c_proj_bias[i],\
        self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],\
        self.mlp_c_proj_weight[i],self.mlp_c_proj_bias[i])
      unif_samples = rand.rand()
      ret = openclk.kernel_1(h,self.ln_f_weight, self.ln_f_bias,self.lm_head_weight,temperature,unif_samples).astype(np.int32)[0]  
      return ret
    else:
      x = openclk.tok_emb(tokens,self.wte_weight,self.wpe_weight,n_tokens)
      for i in range(len(self.h)-1):
        x = openclk.kernel_2(x,self.ln_1_weight[i], self.ln_1_bias[i],self.attn_c_attn_weight[i],self.attn_c_attn_bias[i],self.attn_cache_kv[i],self.attn_c_proj_weight2[i],self.attn_c_proj_bias[i],self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],self.mlp_c_proj_weight_unf[i],self.mlp_c_proj_bias[i],n_tokens,MAX_CONTEXT)
    unif_samples = rand.rand()
    ret = openclk.kernel_3(x,self.ln_1_weight[-1], self.ln_1_bias[-1],self.attn_c_attn_weight[-1],self.attn_c_attn_bias[-1],self.attn_cache_kv[-1]\
    ,self.ln_f_weight, self.ln_f_bias,n_tokens,MAX_CONTEXT,self.lm_head_weight_unf,temperature,unif_samples).astype(np.int32)[0]
    return ret

  def __call__(self, tokens, start_pos, temperature:np.float32=0.0,n_tokens=1):
    return self.forward(tokens, start_pos, temperature,n_tokens)

VOCAB_SIZE = 50257
class GPT2:
  @staticmethod
  def build():
    model = Transformer(n_layers=12,n_heads=12,dim=768,norm_eps=1e-5,vocab_size=VOCAB_SIZE) #small
    return GPT2(model)

  def __init__(self, model):
    self.model = model

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1,expected_tokens=None):
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
      np.testing.assert_equal(tok,expected_tokens[start_pos-n_tokens])#2
      toks.append(tok)
    return decode(toks)

# **** main code ****

if __name__ == "__main__":
  rand = Rand()

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


  MAX_CONTEXT = len(encode(default_prompt))+100
  openclk = opencl_kernels.Opencl_Kernels(dim=768,n_heads=12,max_context=MAX_CONTEXT)
  dim = 768
  n_heads = 12

  filehandler = open("weights_128.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  gpt2.model.convert()
  filehandler = open("new_converted_model_128_3.pickle", 'rb')  
  gpt2_2 = pickle.load(filehandler)

  print(np.shape(gpt2.model.h[0].attn.c_attn.weight),np.shape(gpt2_2.model.attn_c_attn_weight[0]))
  print(type(gpt2.model.h[0].attn.c_attn.weight[0][0]),type(gpt2_2.model.attn_c_attn_weight[0][0][0]))

  #todo get rid of this
  for i in range(12):
    gpt2.model.h[i].attn.c_proj.weight = gpt2_2.model.attn_c_proj_weight[i]
    gpt2.model.h[i].attn.c_proj.weight = np.asfortranarray(gpt2.model.h[i].attn.c_proj.weight)
    gpt2.model.h[i].mlp.c_proj.weight = gpt2_2.model.mlp_c_proj_weight[i]
    gpt2.model.h[i].mlp.c_proj.weight = np.asfortranarray(gpt2.model.h[i].mlp.c_proj.weight)
  
  text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens)
  print((f"Response:", "green"), text)
  rand = Rand()
  MAX_CONTEXT = len(encode("What happened in 1939?"))+100
  openclk = opencl_kernels.Opencl_Kernels(dim=768,n_heads=12,max_context=MAX_CONTEXT)
  text = gpt2.generate(prompt="What happened in 1939?", max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens_b)
  print((f"Response:", "green"), text)

  exit()
  MAX_CONTEXT = len(encode(default_prompt))+100
  openclk = opencl_kernels.Opencl_Kernels(dim=1024,n_heads=16,max_context=MAX_CONTEXT)
  dim = 1024
  n_heads = 16

  filehandler = open("weights_med.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  gpt2.model.convert()

  for i in range(24):
    gpt2.model.h[i].attn.c_attn.weight = np.asfortranarray(gpt2.model.h[i].attn.c_attn.weight)
    gpt2.model.h[i].attn.c_proj.weight = np.asfortranarray(gpt2.model.h[i].attn.c_proj.weight)
    gpt2.model.h[i].mlp.c_fc.weight = np.asfortranarray(gpt2.model.h[i].mlp.c_fc.weight)
    gpt2.model.h[i].mlp.c_proj.weight = np.asfortranarray(gpt2.model.h[i].mlp.c_proj.weight)
  rand = Rand()
  text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens_med)
  print((f"Response:", "green"), text)
  exit()
#!/usr/bin/env python3 #for tinygrad repo, get rid of libs etc
# can I beat https://github.com/jaymody/xpicoGPT.git?
# beating https://github.com/WAUthethird/stupidGPT should be easy
from typing import Union, Tuple
from tqdm import trange
import numpy as np
import os
import pickle
import kernels
from transformers import AutoModelForCausalLM
d = "OpenCL"
folder = ""
try:
   import Metal
   import metal_kernels_large
   d = "Metal"
   print("Using Metal")
except ImportError:
    import pyopencl as cl
    print("Using OpenCL")
    pass
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    d = "CUDA"
except ImportError:
    pass

import transformer

if d == "Metal":
    device = Metal.MTLCreateSystemDefaultDevice()
    queue = device.newCommandQueue()
    params = {"queue":queue,"device":device}
if d == "OpenCL":
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    mf = cl.mem_flags
    params = {"ctx":ctx,"mf":mf,"queue":cl.CommandQueue(ctx)}

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

class Rand:
  def __init__(self):
    self.seed = 420

  def rand(self):
    self.seed += 1
    rng = np.random.default_rng(self.seed)
    rng_np_buffer = rng.random(size=1, dtype=np.float32).astype(dtype=np.float32, copy=False)
    return rng_np_buffer[0]
    
class Transformer:
  def to_buffer(self,n_heads,dim):
    self.n_heads = n_heads
    self.dim = dim

    print("copying ln_1_weight")
    for i in range(len(self.ln_1_weight)):
      self.ln_1_weight[i] = transformer.create_buffer(self.ln_1_weight[i],d,params)

    print("copying ln_1_bias")
    for i in range(len(self.ln_1_weight)):
      self.ln_1_bias[i] = transformer.create_buffer(self.ln_1_bias[i],d,params)

    print("copying attn_c_attn_weight")
    for i in range(len(self.ln_1_weight)):
      self.attn_c_attn_weight[i] = transformer.create_buffer(self.attn_c_attn_weight[i].transpose(1,0).flatten(),d,params)

    print("copying attn_c_attn_bias")
    for i in range(len(self.ln_1_weight)):
      self.attn_c_attn_bias[i] = transformer.create_buffer(self.attn_c_attn_bias[i],d,params)

    print("copying attn_c_proj_bias")
    for i in range(len(self.ln_1_weight)):
      self.attn_c_proj_bias[i] = transformer.create_buffer(self.attn_c_proj_bias[i],d,params)

    print("copying ln_2_weight")
    for i in range(len(self.ln_1_weight)):
      self.ln_2_weight[i] = transformer.create_buffer(self.ln_2_weight[i],d,params)

    print("copying ln_2_bias")
    for i in range(len(self.ln_1_weight)):
      self.ln_2_bias[i] = transformer.create_buffer(self.ln_2_bias[i],d,params)

    print("copying mlp_c_fc_bias")
    for i in range(len(self.ln_1_weight)):
      self.mlp_c_fc_bias[i] = transformer.create_buffer(self.mlp_c_fc_bias[i],d,params)

    print("copying mlp_c_proj_bias")
    for i in range(len(self.ln_1_weight)):
      self.mlp_c_proj_bias[i] = transformer.create_buffer(self.mlp_c_proj_bias[i],d,params)

    print("copying ln_f_weight")
    self.ln_f_weight = transformer.create_buffer(self.ln_f_weight,d,params)
  
    print("copying ln_f_bias")
    self.ln_f_bias = transformer.create_buffer(self.ln_f_bias,d,params)

    print("copying mlp_c_fc_weight")
    for i in range(len(self.ln_1_weight)):
      self.mlp_c_fc_weight[i] = transformer.create_buffer(self.mlp_c_fc_weight[i].transpose(1,0).flatten(),d,params)

    print("copying lm_head_weight_unf")
    self.lm_head_weight_unf = transformer.create_buffer(self.lm_head_weight.transpose(),d,params)

    print("copying lm_head_weight")
    self.lm_head_weight = transformer.create_buffer(self.lm_head_weight.flatten(),d,params)

    print("copying self_wte_weight")
    self.wte_weight = transformer.create_buffer(self.wte_weight.astype(np.float32),d,params)

    print("copying self_wpe_weight")
    self.wpe_weight = transformer.create_buffer(self.wpe_weight,d,params)

    print("creating attn_cache_kv")
    self.attn_cache_kv = []
    for i in range(len(self.ln_1_weight)):
      self.attn_cache_kv.append(transformer.create_buffer_empty(2*MAX_CONTEXT*n_heads*64*4,d,params))

    print("copying attn_c_proj_weight") #TODO
    for i in range(len(self.ln_1_weight)):
      self.attn_c_proj_weight[i] = transformer.create_buffer(self.attn_c_proj_weight[i].flatten(),d,params)

    print("copying mlp_c_proj_weight_unf") #TODO, asfortranarray and transpose work differently on either of machines atm.
    self.mlp_c_proj_weight_unf = []
    for i in range(len(self.ln_1_weight)):
      self.mlp_c_proj_weight_unf.append(transformer.create_buffer(self.mlp_c_proj_weight[i].transpose().flatten(),d,params)) #TODO untested on CUDA
      self.mlp_c_proj_weight[i] = transformer.create_buffer(self.mlp_c_proj_weight[i].flatten(),d,params)

  def forward(self, tokens, start_pos, temperature:float=0.8,n_tokens=444):
    if start_pos > 0:
      h = k.add(self.wte_weight,self.wpe_weight,start_pos,tokens[0])
      attn_dim = self.dim
      for i in range(0,len(self.ln_1_weight)):
        h = k.kernel_0(h,self.ln_1_weight[i],\
        self.ln_1_bias[i],self.attn_c_attn_weight[i],\
        self.attn_c_attn_bias[i],\
        self.attn_cache_kv[i],\
        self.attn_c_proj_weight[i],self.attn_c_proj_bias[i],\
        self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],\
        self.mlp_c_proj_weight[i],self.mlp_c_proj_bias[i],start_pos,attn_dim,i)
      unif_samples = rand.rand()
      ret = k.kernel_1(h,self.ln_f_weight, self.ln_f_bias,self.lm_head_weight,temperature,unif_samples).astype(np.int32)[0]  
      return ret
    else:
      x = k.tok_emb(tokens,self.wte_weight,self.wpe_weight,n_tokens)
      for i in range(len(self.ln_1_weight)-1):
        x = k.kernel_2(x,self.ln_1_weight[i], self.ln_1_bias[i],self.attn_c_attn_weight[i],self.attn_c_attn_bias[i],self.attn_cache_kv[i],self.attn_c_proj_weight[i],self.attn_c_proj_bias[i],self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],self.mlp_c_proj_weight_unf[i],self.mlp_c_proj_bias[i],n_tokens,MAX_CONTEXT,i)
    unif_samples = rand.rand()
    ret = k.kernel_3(x,self.ln_1_weight[-1], self.ln_1_bias[-1],self.attn_c_attn_weight[-1],self.attn_c_attn_bias[-1],self.attn_cache_kv[-1]\
    ,self.ln_f_weight, self.ln_f_bias,n_tokens,MAX_CONTEXT,self.lm_head_weight_unf,temperature,unif_samples).astype(np.int32)[0]
    return ret

  def __call__(self, tokens, start_pos, temperature:np.float32=0.0,n_tokens=1):
    return self.forward(tokens, start_pos, temperature,n_tokens)
  
def delete_buffers(m): #TODO, do this with a loop
    m.wpe_weight.delete()
    m.ln_f_weight.delete()
    m.ln_f_bias.delete()
    m.wte_weight.delete()
    m.lm_head_weight.delete()
    m.lm_head_weight_unf.delete()
    for x in range(len(m.ln_1_bias)): #TODO
      m.mlp_c_proj_bias[x].delete()
      m.mlp_c_proj_weight[x].delete()
      m.mlp_c_proj_weight_unf[x].delete()
      m.mlp_c_fc_bias[x].delete()
      m.mlp_c_fc_weight[x].delete()
      m.attn_c_proj_bias[x].delete()
      m.attn_c_proj_weight[x].delete()
      m.attn_c_attn_bias[x].delete()
      m.attn_c_attn_weight[x].delete()
      m.ln_1_bias[x].delete()
      m.ln_1_weight[x].delete()
      m.ln_2_bias[x].delete()
      m.ln_2_weight[x].delete()
      m.attn_cache_kv[x].delete()

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
      if expected_tokens != None: #TODO REMOVE 13
        np.testing.assert_equal(tok,expected_tokens[start_pos-n_tokens])
      toks.append(tok)
    return decode(toks)


def get_model(model_size):
  gpt2_blank = GPT2(None)
  gpt2_blank.model = Transformer()
  n_layers = {"gpt2":12,"gpt2-medium":24,"gpt2-large":36,"gpt2-xl":48}
  model = AutoModelForCausalLM.from_pretrained(model_size)

  print("converting wpe_weight")
  gpt2_blank.model.wpe_weight = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float32)

  print("converting ln_f.weight")
  gpt2_blank.model.ln_f_weight = model.transformer.ln_f.weight.detach().cpu().numpy().astype(np.float32)

  print("converting ln_f.bias")
  gpt2_blank.model.ln_f_bias = model.transformer.ln_f.bias.detach().cpu().numpy().astype(np.float32)

  print("converting mlp_c_proj.bias")
  gpt2_blank.model.mlp_c_proj_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.mlp_c_proj_bias.append(model.transformer.h[x].mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32))

  print("converting mlp_c_proj.weight")
  gpt2_blank.model.mlp_c_proj_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.mlp_c_proj_weight.append(model.transformer.h[x].mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32))

  print("converting mlp_c_fc.bias")
  gpt2_blank.model.mlp_c_fc_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.mlp_c_fc_bias.append(model.transformer.h[x].mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32))

  print("converting mlp_c_fc.weight")
  gpt2_blank.model.mlp_c_fc_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.mlp_c_fc_weight.append(model.transformer.h[x].mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32))

  print("converting attn_c_proj.bias")
  gpt2_blank.model.attn_c_proj_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.attn_c_proj_bias.append(model.transformer.h[x].attn.c_proj.bias.detach().cpu().numpy().astype(np.float32))

  print("converting attn_c_proj.weight")
  gpt2_blank.model.attn_c_proj_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.attn_c_proj_weight.append(model.transformer.h[x].attn.c_proj.weight.detach().cpu().numpy().astype(np.float32))

  print("converting attn_c_attn.bias")
  gpt2_blank.model.attn_c_attn_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.attn_c_attn_bias.append(model.transformer.h[x].attn.c_attn.bias.detach().cpu().numpy().astype(np.float32))

  print("converting attn_c_attn.weight")
  gpt2_blank.model.attn_c_attn_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.attn_c_attn_weight.append(model.transformer.h[x].attn.c_attn.weight.detach().cpu().numpy().astype(np.float32))

  print("converting ln_1.bias")
  gpt2_blank.model.ln_1_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.ln_1_bias.append(model.transformer.h[x].ln_1.bias.detach().cpu().numpy().astype(np.float32))

  print("converting ln_1.weight")
  gpt2_blank.model.ln_1_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.ln_1_weight.append(model.transformer.h[x].ln_1.weight.detach().cpu().numpy().astype(np.float32))

  print("converting ln_2.bias")
  gpt2_blank.model.ln_2_bias = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.ln_2_bias.append(model.transformer.h[x].ln_2.bias.detach().cpu().numpy().astype(np.float32))

  print("converting ln_2.weight")
  gpt2_blank.model.ln_2_weight = []
  for x in range(n_layers[model_size]):
      gpt2_blank.model.ln_2_weight.append(model.transformer.h[x].ln_2.weight.detach().cpu().numpy().astype(np.float32))
    
  print("converting wte.weight")
  gpt2_blank.model.wte_weight = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)

  print("converting lm_head.weight")
  gpt2_blank.model.lm_head_weight = model.lm_head.weight.detach().cpu().numpy().astype(np.float32).transpose(1,0)

  with open(folder+model_size+".pickle", 'wb') as outp:
      pickle.dump(gpt2_blank, outp)

# **** main code ****

if __name__ == "__main__":
  rand = Rand()

default_prompt = "What is the answer to life, the universe, and everything?"
#default_prompt = "What happened in 1939?"
# should output:
# .... The Jewish people rejected

#(tg random) should output:
#It was a very fateful day.
#When the Nazis occupied Poland in 1939....

np.random.seed(28)

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

#TODO, this is the current, but wrong output
expected_tokens_b = [198, 198, 1026, 373, 257, 845, 46873, 1110, 13, 198, 
198, 2215, 262, 19147, 12030, 12873, 287, 24414, 11, 262, 
6771, 547, 407, 3142, 284, 670, 287, 262, 17590, 11, 
645, 2300, 703, 881, 484, 2227, 284, 13, 383, 1917, 
2627, 1598, 618, 262, 5103, 1664, 286, 262, 309, 9116, 
4623, 268, 4618, 11, 543, 925, 281, 3113, 329, 262, 
11908, 12, 1273, 14414, 41460, 11, 3414, 617, 19008, 284, 
262, 24830, 4572, 13, 198, 198, 464, 4479, 286, 7570, 
21773, 2066, 82, 1908, 734, 11628, 284, 262, 31062, 13, 
1881, 373, 1912, 319, 257, 1080, 286, 2829, 12370, 4827]

expected_tokens_med = [198, 198, 1544, 468, 262, 2694, 290, 262, 481, 284, 3853, 475, 339, 2391,\
  2314, 2222, 2241, 284, 466, 340, 13, 679, 318, 7787, 284, 307, 3436, 290,\
  7787, 284, 2222, 1854, 656, 340, 13, 679, 318, 7787, 284, 307, 33046, 290,\
  7787, 284, 307, 8606, 13, 198, 198, 4864, 11, 339, 318, 407, 3436, 287,\
  465, 3252, 286, 5287, 13, 198, 198, 22210, 4952, 502, 326, 3252, 2125,\
  470, 262, 6808, 2728, 286, 262, 1917, 13, 198, 198, 2025, 37848, 284,\
  674, 10251, 481, 1282, 611, 356, 12553, 262, 4950, 2000, 1176, 356,\
  423, 13, 198, 198, 2215, 345]

expected_tokens_large = [198, 198, 1532, 345, 550, 257, 40663, 11, 345, 561, 
2192, 1382, 340, 656, 262, 6766, 13, 2293, 477, 11, 
345, 714, 655, 4829, 262, 1468, 2272, 18556, 656, 262, 
8137, 290, 1956, 340, 7382, 13, 198, 198, 1537, 326, 
40663, 561, 779, 262, 976, 3716, 1080, 284, 366, 33327, 
1, 262, 3404, 319, 262, 4417, 286, 262, 5440, 13, 
921, 460, 470, 2824, 3404, 416, 17997, 3404, 656, 262, 
1633, 960, 14108, 40663, 318, 517, 588, 257, 16285, 508, 
46561, 262, 3404, 510, 422, 262, 2323, 13, 5455, 11, 
345, 561, 7925, 1657, 12, 6551, 5696, 422, 262, 3668]


a = transformer.create_buffer_empty(1*4,d,params) #TODO can't run medium in isolation without doing this first?

rand = Rand()
MAX_CONTEXT = len(encode(default_prompt))+100
k = kernels.Kernels(dim=768,n_heads=12,max_context=MAX_CONTEXT,device=d)
if os.path.exists(folder+"gpt2.pickle") == False:
  get_model("gpt2")
filehandler = open(folder+"gpt2.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
gpt2.model.to_buffer(12,768)
text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens)
print((f"Response:", "green"), text)
delete_buffers(gpt2.model)
k.save()
exit()

rand = Rand()
k = kernels.Kernels(dim=768,n_heads=12,max_context=MAX_CONTEXT,device=d)
filehandler = open(folder+"gpt2.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
gpt2.model.to_buffer(12,768)
text = gpt2.generate(prompt="What happened in 1939?", max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens_b)
print((f"Response:", "green"), text)
delete_buffers(gpt2.model)
k.save()

MAX_CONTEXT = len(encode(default_prompt))+100
k = kernels.Kernels(dim=1024,n_heads=16,max_context=MAX_CONTEXT,device=d)  
if os.path.exists(folder+"gpt2-medium.pickle") == False:
  get_model(folder+"gpt2-medium")
filehandler = open(folder+"gpt2-medium.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
#gpt2.model.to_buffer2()
gpt2.model.to_buffer(16,1024)
rand = Rand()
text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=None)
print((f"Response:", "green"), text)
delete_buffers(gpt2.model)
k.save()

MAX_CONTEXT = len(encode(default_prompt))+100
dim = 1280
n_heads = 20
rand = Rand()
k = kernels.Kernels(dim=1280,n_heads=20,max_context=MAX_CONTEXT,device=d)
if os.path.exists(folder+"gpt2-large.pickle") == False:
  get_model("gpt2-large")
filehandler = open(folder+"gpt2-large.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
gpt2.model.to_buffer(20,1280)
text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=expected_tokens_large)
print((f"Response:", "green"), text)
delete_buffers(gpt2.model)
k.save()
exit()

'''
if d == "Metal":
  MAX_CONTEXT = len(encode(default_prompt))+100
  dim = 1280
  n_heads = 20
  k = metal_kernels_large.Metal_Kernels(dim=1280,n_heads=20,max_context=MAX_CONTEXT)
  if os.path.exists(folder+"gpt2-large.pickle") == False:
    get_model("gpt2-large")
  filehandler = open(folder+"gpt2-large.pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  gpt2.model.to_buffer(20,1280)
  rand = Rand()
  text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=None)
  print((f"Response:", "green"), text)
  delete_buffers(gpt2.model)
'''
MAX_CONTEXT = len(encode(default_prompt))+100
dim = 1600
n_heads = 25
k = kernels.Kernels(dim=1600,n_heads=25,max_context=MAX_CONTEXT,device=d)
if os.path.exists(folder+"gpt2-xl.pickle") == False:
  get_model("gpt2-xl")
filehandler = open(folder+"gpt2-xl.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
gpt2.model.to_buffer(25,1600)
rand = Rand()
text = gpt2.generate(prompt=default_prompt, max_length=100, temperature=np.float32(0.8), timing=None, batch_size=1,expected_tokens=None)
print((f"Response:", "green"), text)
delete_buffers(gpt2.model)
k.save()
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
   d = "Metal"
   folder = "metal/"
   print("Using Metal")
except ImportError:
    import pyopencl as cl
    print("Using OpenCL")
    pass
import transformer
import argparse, sys

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
    
class Transformer:
  def __init__(self):
    return None

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

      if d == "OpenCL":      
        print("copying attn_c_proj_weight") #TODO
        self.attn_c_proj_weight2 = []
        for i in range(len(self.ln_1_weight)):
          self.attn_c_proj_weight2.append(transformer.create_buffer(np.asfortranarray(self.attn_c_proj_weight[i]),d,params))
          self.attn_c_proj_weight[i] = transformer.create_buffer(self.attn_c_proj_weight[i].flatten(),d,params)

        print("copying mlp_c_proj_weight_unf") #TODO
        self.mlp_c_proj_weight_unf = []
        for i in range(len(self.ln_1_weight)):
          self.mlp_c_proj_weight_unf.append(transformer.create_buffer(np.asfortranarray(self.mlp_c_proj_weight[i]),d,params))
          self.mlp_c_proj_weight[i] = transformer.create_buffer(self.mlp_c_proj_weight[i].flatten(),d,params)
        return
      
      if d == "Metal":
        print("copying attn_c_proj_weight") #TODO
        self.attn_c_proj_weight2 = []
        for i in range(len(self.ln_1_weight)):
          self.attn_c_proj_weight2.append(transformer.create_buffer(np.asfortranarray(self.attn_c_proj_weight[i].transpose()),d,params))
          self.attn_c_proj_weight[i] = transformer.create_buffer(self.attn_c_proj_weight[i],d,params)

        print("copying mlp_c_proj_weight_unf") #TODO
        self.mlp_c_proj_weight_unf = []
        for i in range(len(self.ln_1_weight)):
          self.mlp_c_proj_weight_unf.append(transformer.create_buffer(self.mlp_c_proj_weight[i].transpose(),d,params))
          self.mlp_c_proj_weight[i] = transformer.create_buffer(self.mlp_c_proj_weight[i].flatten(),d,params)

  def forward(self, tokens, start_pos, temperature:float=0.8,n_tokens=444):
    if start_pos > 0:
      h = metalk.add(self.wte_weight,self.wpe_weight,start_pos,tokens[0])
      attn_dim = self.dim
      for i in range(0,len(self.ln_1_weight)):
        h = metalk.kernel_0(h,self.ln_1_weight[i],\
        self.ln_1_bias[i],self.attn_c_attn_weight[i],\
        self.attn_c_attn_bias[i],\
        self.attn_cache_kv[i],\
        self.attn_c_proj_weight[i],self.attn_c_proj_bias[i],\
        self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],\
        self.mlp_c_proj_weight[i],self.mlp_c_proj_bias[i],start_pos,attn_dim,i)
      unif_samples = rand.rand()
      ret = metalk.kernel_1(h,self.ln_f_weight, self.ln_f_bias,self.lm_head_weight,temperature,unif_samples).astype(np.int32)[0]  
      return ret
    else:
      x = metalk.tok_emb(tokens,self.wte_weight,self.wpe_weight,n_tokens)
      for i in range(len(self.ln_1_weight)-1):
        x = metalk.kernel_2(x,self.ln_1_weight[i], self.ln_1_bias[i],self.attn_c_attn_weight[i],self.attn_c_attn_bias[i],self.attn_cache_kv[i],self.attn_c_proj_weight2[i],self.attn_c_proj_bias[i],self.ln_2_weight[i], self.ln_2_bias[i],\
        self.mlp_c_fc_weight[i],self.mlp_c_fc_bias[i],self.mlp_c_proj_weight_unf[i],self.mlp_c_proj_bias[i],n_tokens,MAX_CONTEXT,i)
    unif_samples = rand.rand()
    ret = metalk.kernel_3(x,self.ln_1_weight[-1], self.ln_1_bias[-1],self.attn_c_attn_weight[-1],self.attn_c_attn_bias[-1],self.attn_cache_kv[-1]\
    ,self.ln_f_weight, self.ln_f_bias,n_tokens,MAX_CONTEXT,self.lm_head_weight_unf,temperature,unif_samples).astype(np.int32)[0]
    return ret

  def __call__(self, tokens, start_pos, temperature:np.float32=0.0,n_tokens=1):
    return self.forward(tokens, start_pos, temperature,n_tokens)
  
dims = {"gpt2":768,"gpt2-medium":1024,"gpt2-large":1280}
n_heads = {"gpt2":12,"gpt2-medium":16,"gpt2-large":20}
model_size = "gpt2"

default_prompt = "What is the answer to life, the universe, and everything?"
output_length = 100
temperature = 0.8
parser=argparse.ArgumentParser()
parser.add_argument("--p", help="prompt")
parser.add_argument("--l", help="number of tokens to generate")
parser.add_argument("--t", help="temperature")
parser.add_argument("--m", help="model. gpt2, gpt2-medium or gpt2-large default is gpt2")
args=parser.parse_args()
if args.p is not None: default_prompt = args.p
if args.t is not None: temperature = args.t
if args.l is not None: output_length = int(args.l)
if args.m is not None: model_size = args.m

MAX_CONTEXT = len(encode(default_prompt))+output_length

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
      toks.append(min(50256,tok))
      if tok >= 50256:
         return decode(toks)
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
  a = transformer.create_buffer_empty(1*4,d,params) #TODO can't run medium in isolation without doing this first?
  rand = Rand()
  metalk = kernels.Kernels(dim=dims[model_size],n_heads=n_heads[model_size],max_context=MAX_CONTEXT,device=d)
  if os.path.exists(folder+model_size+".pickle") == False:
    get_model(model_size)
  filehandler = open(folder+model_size+".pickle", 'rb')  
  gpt2 = pickle.load(filehandler)
  gpt2.model.to_buffer(n_heads[model_size],dims[model_size])
  text = gpt2.generate(prompt=default_prompt, max_length=output_length, temperature=temperature, timing=None, batch_size=1,expected_tokens=None)
  print((f"Response:", "green"), text)
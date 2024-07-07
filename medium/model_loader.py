'''
import os, json, pathlib, zipfile, pickle, tarfile, struct
from tqdm import tqdm
from typing import Dict, Union, List, Optional, Any, Tuple
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, fetch
from tinygrad.shape.view import strides_for_shape

def torcload(fn:str) -> Dict[str, Tensor]:
  """
  Loads a torch .pth file from disk.

  ```python
  state_dict = nn.state.torcload("test.pth")
  ```
  """
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")

  offsets: Dict[Union[str, int], int] = {}
  lens: Dict[Union[str, int], int] = {}
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    byte_offset = offsets[storage[2]]+storage_offset*storage[1].itemsize
    ret = t[byte_offset:byte_offset+prod(size)*storage[1].itemsize].bitcast(storage[1])

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      assert storage[1] != dtypes.bfloat16, "can't CLANG permute BF16"
      # TODO: find a nice way to support all shapetracker on disktensors
      # TODO: BUG: a ".realize()" is needed here for 'GPU=1 python3 test/models/test_efficientnet.py TestEfficientNet.test_car'
      ret = ret.clang().reshape(intermediate_shape).permute(permute_indexes).realize()

    return ret.reshape(size).numpy()

  class Parameter:
    def __setstate__(self, state): self.tensor = state[0]

  deserialized_objects: Dict[str, Any] = {}
  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16, "IntStorage": dtypes.int32,
               "LongStorage": dtypes.int64, "_rebuild_tensor_v2": _rebuild_tensor_v2, "FloatTensor": None, "Parameter": Parameter}
  whitelist = {"torch", "collections", "numpy", "_codecs"}  # NOTE: this is not for security, only speed
  class Dummy: pass
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      module_root = module.split(".")[0]
      if module_root not in whitelist:
        if DEBUG >= 2: print(f"WARNING: returning Dummy for {module} {name}")
        return Dummy
      return intercept[name] if module_root == "torch" else super().find_class(module, name)
    def persistent_load(self, pid): return deserialized_objects.get(pid, pid)

  if zipfile.is_zipfile(fn):
    myzip = zipfile.ZipFile(fn, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n) as myfile:
          offsets[n.split("/")[-1]] = myfile._orig_compress_start # type: ignore
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  elif tarfile.is_tarfile(fn):
    with tarfile.open(fn, "r") as tar:
      storages_offset = tar.getmember('storages').offset_data
      f = unwrap(tar.extractfile('storages'))
      for i in range(TorchPickle(f).load()):  # num_storages
        (key, _, storage_type), sz = TorchPickle(f).load(), struct.unpack('<q', f.read(8))[0]
        offsets[key] = storages_offset + f.tell()
        f.seek(sz*storage_type.itemsize, 1)
      f = unwrap(tar.extractfile('tensors'))
      for _ in range(TorchPickle(f).load()):  # num_tensors
        (key, storage_id, _), ndim, _ = TorchPickle(f).load(), struct.unpack('<i', f.read(4))[0], f.read(4)
        size, stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim)), struct.unpack(f'<{ndim}q', f.read(8 * ndim))
        storage_offset = struct.unpack('<q', f.read(8))[0]
        deserialized_objects[str(key)] = _rebuild_tensor_v2((None, storage_type, storage_id, None, -1), storage_offset, size, stride)
      return {k:v.tensor if isinstance(v, Parameter) else v for k,v in TorchPickle(unwrap(tar.extractfile('pickle'))).load().items()}
  else:
    with open(fn, "rb") as f:
      pkl = TorchPickle(f)
      _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), f.tell(), pkl.load(), pkl.load(), f.tell()
      for i in ids:
        offsets[i] = base_offset + 8
        base_offset += 8 + lens[i]
      f.seek(rwd)
      return TorchPickle(f).load()

model_size = "gpt2"
weights = torcload(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorcmodel.bin'))
print("weights =",weights)
print("done")

'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import numpy as np

class GPT2:
    def __init__(self): return None
class Transformer:
    def __init__(): return None
class Transformer:
    def __init__(self): return None
class Embedding_2:
    def __init__(self): return None
class Embedding:
    def __init__(self): return None
class TransformerBlock:
    def __init__(): return None
class Attention:
    def __init__(): return None
class Linear:
    def __init__(): return None
class FeedForward:
    def __init__(): return None
class LayerNorm:
    def __init__(): return None


def convert_2d(a,b):
    for i in range(a.size()[0]):
        for j in range(a.size()[1]):
            b[i][j] = np.float32(a[i][j].item())
    return

def convert_1d(a,b):
    for i in range(a.size()[0]):
        b[i] = np.float32(a[i].item())
    return


filehandler = open("weights_128_converted.pickle", 'rb')  
gpt2 = pickle.load(filehandler)
print(type(gpt2))

#blank gpt2, no values
#only add values from huggingface's
#add values until equal to converted model
gpt2_blank = GPT2()
gpt2_blank.model = Transformer()

print(gpt2.model.h[0].ln_1.weight[1])

model = AutoModelForCausalLM.from_pretrained("gpt2")
print(model)
print(model.transformer.wte.weight)
print(type(model.transformer.wte.weight))
print(model.transformer.wte.weight.size())
print(model.transformer.wte.weight[0][0].item())

#x = model.transformer.wpe.weight.detach().cpu().numpy()
#print(x)
#print(type(x))
#print(type(x[0][0]))

print("converting wpe_weight")
gpt2_blank.model.wpe_weight = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float32)

print("converting ln_f.weight")
gpt2_blank.model.ln_f_weight = model.transformer.ln_f.weight.detach().cpu().numpy().astype(np.float32)

print("converting ln_f.bias")
gpt2_blank.model.ln_f_bias = model.transformer.ln_f.bias.detach().cpu().numpy().astype(np.float32)

print("converting mlp_c_proj.bias")
gpt2_blank.model.mlp_c_proj_bias = []
for x in range(12):
    gpt2_blank.model.mlp_c_proj_bias.append(model.transformer.h[x].mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32))

print("converting mlp_c_proj.weight")
gpt2_blank.model.mlp_c_proj_weight = []
for x in range(12):
    gpt2_blank.model.mlp_c_proj_weight.append(model.transformer.h[x].mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32))


print("converting mlp_c_fc.bias")
gpt2_blank.model.mlp_c_fc_bias = []
for x in range(12):
    gpt2_blank.model.mlp_c_fc_bias.append(model.transformer.h[x].mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32))

print("converting mlp_c_fc.weight")
gpt2_blank.model.mlp_c_fc_weight = []
for x in range(12):
    gpt2_blank.model.mlp_c_fc_weight.append(model.transformer.h[x].mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32))

print("converting attn_c_proj.bias")
gpt2_blank.model.attn_c_proj_bias = []
for x in range(12):
    gpt2_blank.model.attn_c_proj_bias.append(model.transformer.h[x].attn.c_proj.bias.detach().cpu().numpy().astype(np.float32))

print("converting attn_c_proj.weight")
gpt2_blank.model.attn_c_proj_weight = []
for x in range(12):
    gpt2_blank.model.attn_c_proj_weight.append(model.transformer.h[x].attn.c_proj.weight.detach().cpu().numpy().astype(np.float32))

print("converting attn_c_attn.bias")
gpt2_blank.model.attn_c_attn_bias = []
for x in range(12):
    gpt2_blank.model.attn_c_attn_bias.append(model.transformer.h[x].attn.c_attn.bias.detach().cpu().numpy().astype(np.float32))

print("converting attn_c_attn.weight")
gpt2_blank.model.attn_c_attn_weight = []
for x in range(12):
    gpt2_blank.model.attn_c_attn_weight.append(model.transformer.h[x].attn.c_attn.weight.detach().cpu().numpy().astype(np.float32))

print("converting ln_1.bias")
gpt2_blank.model.ln_1_bias = []
for x in range(12):
    gpt2_blank.model.ln_1_bias.append(model.transformer.h[x].ln_1.bias.detach().cpu().numpy().astype(np.float32))

print("converting ln_1.weight")
gpt2_blank.model.ln_1_weight = []
for x in range(12):
    gpt2_blank.model.ln_1_weight.append(model.transformer.h[x].ln_1.weight.detach().cpu().numpy().astype(np.float32))

print("converting ln_2.bias")
gpt2_blank.model.ln_2_bias = []
for x in range(12):
    gpt2_blank.model.ln_2_bias.append(model.transformer.h[x].ln_2.bias.detach().cpu().numpy().astype(np.float32))

print("converting ln_2.weight")
gpt2_blank.model.ln_2_weight = []
for x in range(12):
    gpt2_blank.model.ln_2_weight.append(model.transformer.h[x].ln_2.weight.detach().cpu().numpy().astype(np.float32))
   

print("converting wte.weight")
gpt2_blank.model.wte_weight = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)


print("converting lm_head.weight")
print(model.lm_head.weight)
gpt2_blank.model.lm_head_weight = model.lm_head.weight.detach().cpu().numpy().astype(np.float32).transpose(1,0)


with open('new_converted_model_128_3.pickle', 'wb') as outp:
    pickle.dump(gpt2_blank, outp)

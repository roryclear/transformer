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


gpt2_blank = GPT2()
gpt2_blank.model = Transformer()

model_size = "gpt2-medium"
n_layers = {"gpt2":12,"gpt2-medium":24}
model = AutoModelForCausalLM.from_pretrained(model_size)
print(model)
print(model.transformer.wte.weight)
print(type(model.transformer.wte.weight))
print(model.transformer.wte.weight.size())
print(model.transformer.wte.weight[0][0].item())

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
print(model.lm_head.weight)
gpt2_blank.model.lm_head_weight = model.lm_head.weight.detach().cpu().numpy().astype(np.float32).transpose(1,0)


with open('new_converted_model_medium.pickle', 'wb') as outp:
    pickle.dump(gpt2_blank, outp)

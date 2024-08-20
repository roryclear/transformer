import numpy as np
import time
import math
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except ImportError:
    pass
try:
   import Metal
except ImportError:
    pass
import transformer
import pyopencl as cl
import pickle
import os

ls = 256

kernel_prefix = {"OpenCL":"","Metal":"#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n","CUDA":""}
uint3_arg = {"OpenCL":"","Metal":", uint3 gid [[thread_position_in_grid]]","CUDA":""}
func_dec = {"OpenCL":"__kernel","Metal":"kernel","CUDA":"__global__"} #TODO vs local cuda?
var_dec = {"OpenCL":"__global","Metal":"device","CUDA":""}
barrier = {"OpenCL":"barrier(CLK_LOCAL_MEM_FENCE);","Metal":"threadgroup_barrier(mem_flags::mem_threadgroup);","CUDA":" __syncthreads();"}
global_idx = {"OpenCL":"get_global_id(0)","Metal":"gid.x","CUDA":"threadIdx.x+blockIdx.x*blockDim.x"}
local_var = {"OpenCL":"__attribute__ ((aligned (16))) __local","Metal":"threadgroup","CUDA":"__shared__"}

class Kernels:
    def __init__(self,dim,n_heads,max_context,device):
        self.d = device
        self.prg_cache = {}
        if os.path.exists("ls_cache.pickle") == True:
            self.ls_cache = pickle.load(open("ls_cache.pickle", 'rb'))
        else:
            self.ls_cache = {}
        if os.path.exists("prg_str_cache.pickle") == True:
            self.prg_str_cache = pickle.load(open("prg_str_cache.pickle", 'rb'))
        else:
            self.prg_str_cache = {}

        self.dim = dim
        self.n_heads = n_heads
        self.max_context = max_context
        if device == "Metal":
            self.device = Metal.MTLCreateSystemDefaultDevice()
            self.queue = self.device.newCommandQueue()
            self.params = {"queue":self.queue,"device":self.device}
        if device == "OpenCL":
            platform = cl.get_platforms()
            my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
            ctx = cl.Context(devices=my_gpu_devices)
            self.queue = cl.CommandQueue(ctx)
            mf = cl.mem_flags
            prg = None
            self.params = {"ctx":ctx,"mf":mf,"queue":self.queue}
        if device == "CUDA":
            self.params = None
        

        for k in self.prg_str_cache.keys():
            self.prg_cache[k] = transformer.compile(self.prg_str_cache[k],self.d,self.params)
    
    def save_cache(self):
        with open("ls_cache.pickle", 'wb') as outp:
            pickle.dump(self.ls_cache, outp)
        with open("prg_str_cache.pickle", 'wb') as outp:
            pickle.dump(self.prg_str_cache, outp)


    def add(self,a_g,b_g,b_s=0,a_s=0):
        if hasattr(self, 'add_res_g') == False:
            self.add_res_g = transformer.create_buffer_empty(self.dim*4,self.d,self.params)
        prg_str = f"""
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void add(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} const float *b, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
        int gidx0 = {global_idx[self.d]};
        if(gidx0 < {self.dim}) {{
            res[gidx0] = a[{int(a_s)*self.dim} + gidx0] + b[gidx0 + {b_s*self.dim}];
        }}   
        }}
        """
        prg = transformer.compile(prg_str,self.d,self.params)
        transformer.run(prg,"add",self.params,[a_g, b_g,self.add_res_g],self.dim,ls,self.d)
        return self.add_res_g
    
    def test_ls(self,prg_str,func,args,gs,start_pos=None):
        if func not in self.ls_cache:
            ls = 16
            if func[0:7] == "k0_mm3_":
                ls = math.ceil(self.n_heads/16)*16 #TODO, self.n_heads > 16 breaks it
            og_output = []
            f_ls = None
            f_t = None
            f_prg = None
            args_copy = []
            for a in args: args_copy.append(a.copy(self.params))
            while ls <= 1024:
                if og_output == []:
                    temp = ls
                    ls = 256
                    prg_str_e = eval(f'f"""{prg_str}"""')
                    library = transformer.compile(prg_str_e,self.d,self.params)
                    transformer.run(library,func,self.params,args_copy,gs,ls,self.d)
                    for a in args_copy: og_output.append(np.copy(a.np()))
                    ls = temp
                    for a in args_copy: transformer.buffer.delete(a)
                    args_copy = []
                    for a in args: args_copy.append(a.copy(self.params))
                t = 0
                for _ in range(1):
                    #print(func)
                    temp = ls
                    prg_str_e = eval(f'f"""{prg_str}"""')
                    library = transformer.compile(prg_str_e,self.d,self.params)
                    t += transformer.run(library,func,self.params,args_copy,gs,ls,self.d,True)
                    '''
                    for i in range(len(args_copy)):
                        print("testing arg",i,func,ls)
                        if np.allclose(x:= args_copy[i].np(),og_output[i],rtol=1e-4) == False:
                            w = 0
                            for j in range(len(x)):
                                if(np.isclose(x[j],og_output[i][j],rtol=1e-4) == False):
                                    print(j,x[j],og_output[i][j])
                                    w+=1
                            if w > 4: np.testing.assert_allclose(args_copy[i].np(),og_output[i],rtol=1e-4)
                    '''
                    ls = temp
                    if f_t is None or t < f_t:
                        f_t = t
                        f_ls = ls
                        f_prg = prg_str_e
                ls+=16
                #for a in args_copy: transformer.buffer.delete(a)
                #args_copy = []
                #for a in args: args_copy.append(a.copy(self.params))
            for a in args_copy: transformer.buffer.delete(a)
            og_output = []
            #print(func,f_ls,f_t)
            self.ls_cache[func] = f_ls
            self.prg_str_cache[func] = f_prg
        ls = self.ls_cache[func]
        prg_str_e = self.prg_str_cache[func]
        if prg_str_e not in self.prg_cache:
            self.prg_cache[func] = transformer.compile(prg_str_e,self.d,self.params)
        prg = self.prg_cache[func]
        transformer.run(prg,func,self.params,args,gs,self.ls_cache[func],self.d)
        ls = 256

    def tok_emb(self,tokens,weight_g,weight_2_g,no_tokens):
        tokens_g = transformer.create_buffer(tokens.astype(np.int32),self.d,self.params)
        tok_emb_g = transformer.create_buffer_empty(no_tokens*self.dim*4,self.d,self.params)
        prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} int *tokens, {var_dec[self.d]} const float *weight, {var_dec[self.d]} const float *weight2,  {var_dec[self.d]} float *tok_emb{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            int i = gidx0 / {self.dim};
            int j = gidx0 % {self.dim};
            tok_emb[i*{self.dim} + j] = weight[tokens[i]*{self.dim} + j] + weight2[i*{self.dim} + j];
        }}
        """
        #self.test_ls(prg_str,"tok_emb_"+str(self.dim),[tokens_g,weight_g,weight_2_g,tok_emb_g],no_tokens*self.dim)
        #prg_str_e = eval(f'f"""{prg_str}"""')##HERE??
        func = "tok_emb"
        prg_str_e = eval(f'f"""{prg_str}"""')
        library = transformer.compile(prg_str_e,self.d,self.params)
        transformer.run(library,"tok_emb",self.params,[tokens_g,weight_g,weight_2_g,tok_emb_g],no_tokens*self.dim,ls,self.d)
        return tok_emb_g

    def kernel_1(self,h_g,weight_g,bias_g,weight2_g,temperature,random_num):
        if hasattr(self, 'logits_g') == False:
            self.logits_g = transformer.create_buffer_empty(50257*4,self.d,self.params)
        if hasattr(self, 'res') == False:
            self.res = np.zeros(1).astype(np.float32)
        if hasattr(self, 'res_g') == False:
            self.res_g = transformer.create_buffer_empty(1*4,self.d,self.params)
        prg_str = f"""
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void k1_mm4(
            {var_dec[self.d]} float *h, {var_dec[self.d]} const float *weight, {var_dec[self.d]} const float *bias{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            {local_var[self.d]} float mean;
            int lidx0 = {global_idx[self.d]};
            float total = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                total += h[i];
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {self.dim};  
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                h[i] -= mean;
            }}
            {barrier[self.d]}
            total = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                total += pow(h[i],2);
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {self.dim} + 1e-5,0.5);
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                h[i] = (h[i] * weight[i]) / mean + bias[i];
            }}
        }}
        {func_dec[self.d]} void k1_matvec(
            {var_dec[self.d]} const float *h, {var_dec[self.d]} const float *weight2 , {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            res[gidx0] = 0;
            for(int j = 0; j < {self.dim}; j++) {{
                res[gidx0] += h[j] * weight2[gidx0 + j*50257];
            }}
            res[gidx0] /= {temperature};
        }}
        {func_dec[self.d]} void k1_mm5(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}
        {func_dec[self.d]} void k1_mm6(
        {var_dec[self.d]} float *a, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{  
            int gidx0 = {global_idx[self.d]};
            a[gidx0] = exp(a[gidx0] - res[0]);
        }}
        {func_dec[self.d]} void k1_mm8(
        {var_dec[self.d]} float *a, {var_dec[self.d]} const float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            a[gidx0] = a[gidx0] / res[0];
        }}
        {func_dec[self.d]} void k1_mm9( //TODO, doesn't work when ls < 16?
        {var_dec[self.d]} const float *a, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{ 
            {local_var[self.d]} float temp[{ls}];
            int lidx0 = {global_idx[self.d]};
            temp[lidx0] = 0;
            for(int i = lidx0*{math.ceil(50257 / ls)}; i < min(lidx0*{math.ceil(50257 / ls)}+{math.ceil(50257 / ls)},50257); i++) {{
                temp[lidx0] += a[i];
            }}
            {barrier[self.d]}
            float t = 0;
            if(lidx0==0) {{
                for(int i = 0; i < {ls}; i++) {{
                    t+=temp[i];
                }}
            res[0] = t;
            }}     
        }}
        {func_dec[self.d]} void k1_mm10(
        {var_dec[self.d]} float *a{uint3_arg[self.d]})
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}
        """

        if prg_str not in self.prg_cache:
            library = transformer.compile(prg_str,self.d,self.params)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]

        prg_str = f"""
        {func_dec[self.d]} void k1_mm11(
        {var_dec[self.d]} float *a{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(a[gidx0] < {random_num}) {{ //TODO, used to be (a[gidx0] / a[50256])/{random_num}
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}
        """
        prg2 = transformer.compile(prg_str,self.d,self.params) #TODO just pass in random num instead of compiling
        gs =  50257
        transformer.run(prg,"k1_mm4",self.params,[h_g, weight_g, bias_g],1,ls,self.d)
        transformer.run(prg,"k1_matvec",self.params,[h_g, weight2_g,self.logits_g],gs,ls,self.d)
        transformer.run(prg,"k1_mm5",self.params,[self.logits_g,self.res_g],1,1,self.d)
        transformer.run(prg,"k1_mm6",self.params,[self.logits_g,self.res_g],gs,ls,self.d)
        transformer.run(prg,"k1_mm5",self.params,[self.logits_g,self.res_g],1,1,self.d)
        transformer.run(prg,"k1_mm8",self.params,[self.logits_g,self.res_g],gs,ls,self.d)
        transformer.run(prg,"k1_mm9",self.params,[self.logits_g,self.res_g],1,ls,self.d)
        transformer.run(prg,"k1_mm8",self.params,[self.logits_g,self.res_g],gs,ls,self.d)
        transformer.run(prg,"k1_mm10",self.params,[self.logits_g],1,1,self.d)
        transformer.run(prg2,"k1_mm11",self.params,[self.logits_g],gs,ls,self.d)
        transformer.run(prg,"k1_mm9",self.params,[self.logits_g,self.res_g],1,ls,self.d)
        return self.res_g.np(self.params)

    def kernel_3(self,x_g,weight_g,bias_g,attn_weight_g,attn_bias_g,new_cache_g\
        ,ln_f_weight_g,ln_f_bias_g,n_tokens,max_content,lm_head_weight_g,temperature,random_num):
        x0_g = transformer.create_buffer_empty(n_tokens*self.dim*4,self.d,self.params)
        logits_g = transformer.create_buffer_empty(50257*4,self.d,self.params)
        c_g = transformer.create_buffer_empty(max_content*self.dim*3*4,self.d,self.params) #todo, can this be smaller?
        res_g = transformer.create_buffer_empty(1*4,self.d,self.params)
        prg_str = f"""
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void k3_mm({var_dec[self.d]} const float *x_in,
            {var_dec[self.d]} float *x, {var_dec[self.d]} const float *weight, {var_dec[self.d]} const float *bias{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            {local_var[self.d]} float temp2[{n_tokens}];
            int gidx0 = {global_idx[self.d]};
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; 
            temp2[r] = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r+lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r+{self.dim}); i++) {{
                x[i] = x_in[i];
                temp2[r] += x[i];
            }}
            temp[lidx0] = temp2[r];
            {barrier[self.d]}
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {self.dim};  
            }}
            {barrier[self.d]}
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r+lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r+{self.dim}); i++) {{
                x[i] -= temp2[r];
            }}
            {barrier[self.d]}
            temp2[r] = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r+lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r+{self.dim}); i++) {{
                temp2[r] += pow(x[i],2);
            }}
            temp[lidx0] = temp2[r];
            {barrier[self.d]}
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {self.dim} + 1e-5,0.5);
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                x[{self.dim}*r + i] = (x[{self.dim}*r + i] * weight[i]) / temp2[r] + bias[i ];
            }}
        }}
        {func_dec[self.d]} void k3_mm2(
            {var_dec[self.d]} const float *x, {var_dec[self.d]} const float *attn_weight, {var_dec[self.d]} const float *attn_bias,{var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < {self.dim*3*n_tokens}) {{ //TODO
            int gidx0 = {global_idx[self.d]};
            int i = gidx0 / {n_tokens};
            int y = gidx0 % {n_tokens};
            float total = 0;
            for(int k = 0; k < {self.dim}; k++) {{
                total += x[y*{self.dim} + k] * attn_weight[i*{self.dim} + k]; 
            }}
            res[y*{self.dim*3} + i] = total + attn_bias[i];
            }}
        }}
         {func_dec[self.d]} void k3_mm4(
            {var_dec[self.d]} const float *xqkv, {var_dec[self.d]} float *new_cache{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < {self.dim*3*n_tokens}) {{
            int gidx0 = {global_idx[self.d]};
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}];
            new_cache[{max_content*self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}*2];
            }}
        }}
        {func_dec[self.d]} void k3_mm5(
            {var_dec[self.d]} float *x, {var_dec[self.d]} const float *ln_f_weight, {var_dec[self.d]} const float *ln_f_bias{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            {local_var[self.d]} float mean;
            int lidx0 = {global_idx[self.d]};
            float total = 0;
            for(int i = 0; i < {math.ceil(self.dim/ls)}; i++) {{
                total += x[lidx0*{math.ceil(self.dim/ls)} + i];
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {self.dim};  
            }}
            {barrier[self.d]}
            for(int i = 0; i < {math.ceil(self.dim/ls)}; i++) {{
                x[i + lidx0*{math.ceil(self.dim/ls)} + {(n_tokens - 1)*self.dim}] -= mean;
            }}
            {barrier[self.d]}
            total = 0;
            for(int i = 0; i < {math.ceil(self.dim/ls)}; i++) {{
                total += pow(x[lidx0*{math.ceil(self.dim/ls)} + i + {(n_tokens - 1)*self.dim}],2);
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {self.dim} + 1e-5,0.5);
            }}
            {barrier[self.d]}
            for(int i = 0; i < {math.ceil(self.dim/ls)}; i++) {{
                x[i + lidx0*{math.ceil(self.dim/ls)}] = (x[i + lidx0*{math.ceil(self.dim/ls)} + {(n_tokens - 1)*self.dim}] * ln_f_weight[i + lidx0*{math.ceil(self.dim/ls)}]) / mean + ln_f_bias[i + lidx0*{math.ceil(self.dim/ls)}];
            }}
        }}    
        {func_dec[self.d]} void k3_matmul(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} const float *b, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < {50257}) {{
                int x = {global_idx[self.d]};
                float total = 0;
                for(int k = 0; k < {self.dim}; k++) {{
                    total += a[k] * b[x*{self.dim} + k]; 
                }}
                res[x] = total / {temperature}; 
            }}
        }}
        {func_dec[self.d]} void k3_mm6(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        {func_dec[self.d]} void k3_mm7(
        {var_dec[self.d]} float *a, {var_dec[self.d]} const float *res{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < 50257) {{ //TODO
            int gidx0 = {global_idx[self.d]};
            a[gidx0] = exp(a[gidx0] - res[0]);
            }}
        }}
        {func_dec[self.d]} void k3_mm9(
        {var_dec[self.d]} float *a, {var_dec[self.d]} const float *res{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < 50257) {{
            int gidx0 = {global_idx[self.d]};
            a[gidx0] = a[gidx0] / res[0];
            }}
        }}
        {func_dec[self.d]} void k3_mm10(
        {var_dec[self.d]} const float *a, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            int lidx0 = {global_idx[self.d]};
            float t = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},50257); i++) {{
                t += a[i];
            }}
            temp[lidx0] = t;
            {barrier[self.d]}
            if(lidx0 == 0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                res[0] = t;
            }}
        }}
        {func_dec[self.d]} void k3_mm11(
        {var_dec[self.d]} float *a{uint3_arg[self.d]})
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}
        {func_dec[self.d]} void k3_mm12(
        {var_dec[self.d]} float *a{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(a[gidx0] < {random_num}) {{ //TODO, used to be (a[gidx0] / a[50256])/{random_num}
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}
        """
        prg = transformer.compile(prg_str,self.d,self.params)
        transformer.run(prg,"k3_mm",self.params,[x_g, x0_g, weight_g, bias_g],n_tokens*ls,ls,self.d)
        gs = self.dim*3*n_tokens
        transformer.run(prg,"k3_mm2",self.params,[x0_g, attn_weight_g,attn_bias_g,c_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm4",self.params,[c_g, new_cache_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm5",self.params,[x_g, ln_f_weight_g, ln_f_bias_g],1,ls,self.d)
        transformer.run(prg,"k3_matmul",self.params,[x_g, lm_head_weight_g,logits_g],50257,ls,self.d)
        transformer.run(prg,"k3_mm6",self.params,[logits_g,res_g],1,1,self.d)
        gs = 50257
        transformer.run(prg,"k3_mm7",self.params,[logits_g,res_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm6",self.params,[logits_g,res_g],1,1,self.d)
        transformer.run(prg,"k3_mm9",self.params,[logits_g,res_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm10",self.params,[logits_g,res_g],1,ls,self.d)
        transformer.run(prg,"k3_mm9",self.params,[logits_g,res_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm11",self.params,[logits_g],1,1,self.d)
        transformer.run(prg,"k3_mm12",self.params,[logits_g],gs,ls,self.d)
        transformer.run(prg,"k3_mm10",self.params,[logits_g,res_g],1,ls,self.d)
        return res_g.np(self.params)

    def kernel_0(self,a_g,c_g,d_g,e_g,xqkv_g,keys_values_g,attn_c_proj_weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,start_pos,g,j=0):
        if hasattr(self, 'temp_g') == False:
            self.temp_g = transformer.create_buffer_empty(self.n_heads*self.max_context*4,self.d,self.params)
        if hasattr(self, 'xq_temp_g') == False:
            self.xq_temp_g = transformer.create_buffer_empty((self.dim*(self.max_context+1))*4,self.d,self.params) #TODO can this be smaller?

        k0_mm_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} float *a,
            {var_dec[self.d]} float *mean{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            int lidx0 = {global_idx[self.d]};
            float t = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                t += a[i];
            }}
            temp[lidx0] = t;
            {barrier[self.d]}
            if(lidx0==0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                mean[0] = t / {self.dim};  
            }}
            {barrier[self.d]}
            t = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                a[i] -= mean[0];
                t += pow(a[i],2);
            }}
            temp[lidx0] = t;
            {barrier[self.d]}
            if(lidx0==0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                mean[0] = pow(t / {self.dim} + 1e-5,0.5);
            }}
        }}
        """
        k0_mm4_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} float *a,
            {var_dec[self.d]} const float *weight2, {var_dec[self.d]} const float *bias2,
            {var_dec[self.d]} const float *weight3, {var_dec[self.d]} const float *bias3,
            {var_dec[self.d]} float *mean,
            {var_dec[self.d]} float *h, {var_dec[self.d]} float *bias3_temp{uint3_arg[self.d]})
        {{
            //tanh(val0) = ((2.0f*(1/(exp2((val0*(-2.885390043258667f)))+1.0f)))+(-1.0f));
            int gidx0 = {global_idx[self.d]};
            int lidx0 = gidx0 % {ls};
            int i = gidx0 / {ls};
            float total = bias3[i + lidx0*{math.ceil(self.dim*4 / ls)}];
            for(int j = 0; j < {self.dim}; j++) {{
                total += ((h[j] * weight2[j]) / mean[0] + bias2[j]) * weight3[(i + lidx0*{math.ceil(self.dim*4 / ls)})*{self.dim} + j];
            }}
            bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls)}] = 0.5 * total\
            * (1 + ((2.0f*(1/(exp2((total * 0.7978845608 * (1 + 0.044715 * pow(total,2))*(-2.885390043258667f)))+1.0f)))+(-1.0f)));
        }}   
        """

        k0_mm5_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} float *a,
            {var_dec[self.d]} const float *weight4,{var_dec[self.d]} const float *bias4,
            {var_dec[self.d]} float *h_temp, {var_dec[self.d]} float *bias3_temp{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            {local_var[self.d]} float bias4_temp[{self.dim*3}];
            if(gidx0 < {self.dim}) {{
            bias4_temp[gidx0] = bias4[gidx0];
            for(int j = 0; j < {self.dim*4}; j++) {{ 
                bias4_temp[gidx0] += bias3_temp[j] * weight4[gidx0 + j*{self.dim}];
            }}
            a[gidx0] = bias4_temp[gidx0] + h_temp[gidx0];
            }}
        }}
        """

        k0_mm1_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} const float *c, {var_dec[self.d]} const float *d, {var_dec[self.d]} const float *e,
            {var_dec[self.d]} const float *xqkv, {var_dec[self.d]} float *keys_values,
            {var_dec[self.d]} float *xq_temp, {var_dec[self.d]} float *mean{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            int lidx0 = gidx0 % {ls};
            int i = gidx0 / {ls};
            float t = 0;
            int j = lidx0*{math.ceil(self.dim*3 / ls)} + i;
            for(int k = 0; k < {self.dim}; k++) {{
                t += ((a[k] * c[k]) / mean[0] + d[k]) * e[j*{self.dim} + k];
            }}
            if(lidx0*{math.ceil(self.dim*3 / ls)} + i < {self.dim*3}) {{
                xq_temp[j] = xqkv[j];
                if(j < {self.dim}) {{
                    xq_temp[j] += t;
                }}
            }}
            if(j >= {self.dim}) {{
                if(j < {self.dim*2}) {{
                    keys_values[{start_pos}*{self.dim} + j - {self.dim}] = xqkv[j] + t;
                }}
                if(j >= {2*self.dim} && j < {self.dim*3}) {{
                    keys_values[{self.dim*self.max_context} + {start_pos}*{self.dim} + j - {2*self.dim}] = xqkv[j] + t;
                }}
            }}
        }}
        """
        k0_mm2_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} const float *keys_values, {var_dec[self.d]} float *temp3, {var_dec[self.d]} const float *xq_temp{uint3_arg[self.d]})
        {{
            if({global_idx[self.d]} < {self.n_heads*(start_pos+1)*(start_pos+1)}) {{
            int lidx0 = {global_idx[self.d]};
            int x = (lidx0) % {start_pos+1};
            int k = (lidx0) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xq_temp[i + 64*k] * keys_values[x*{self.n_heads*64} + i + 64*k];
            }}          
            if(x + k*{start_pos+1} < {self.n_heads*self.max_context}) {{      
                temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(64)
            }}
            }}
        }}
        """
        k0_mm3_prg_str = """
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void {func}(
            {var_dec[self.d]} float *a,
            {var_dec[self.d]} float *keys_values,
            {var_dec[self.d]} const float *weight,{var_dec[self.d]} const float *bias,
            {var_dec[self.d]} float *temp3, {var_dec[self.d]} float *xq_temp, {var_dec[self.d]} float *mean,
            {var_dec[self.d]} float *h_temp, {var_dec[self.d]} float *h{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            int lidx0 = {global_idx[self.d]};
            if(lidx0 < {self.n_heads}){{
                float m = -INFINITY;
                for(int i = lidx0*{start_pos+1}; i < min(lidx0*{start_pos+1}+{start_pos+1},{self.n_heads*self.max_context}); i++) {{
                    m = max(m,temp3[i]);
                }}
                float t = 0;
                for(int i = 0; i < {start_pos+1}; i++) {{
                    temp3[i + lidx0*{start_pos+1}] = exp(temp3[i + lidx0*{start_pos+1}] - m);
                    t += temp3[i + lidx0*{start_pos+1}];
                }}
                for(int i = 0; i < {start_pos+1}; i++) {{
                    temp3[i + lidx0*{start_pos+1}] /= t;
                }}
            }}
            {barrier[self.d]}
            for(int j = lidx0*{math.ceil(self.dim/ls)}; j < min((lidx0+1)*{math.ceil(self.dim/ls)},{self.dim}); j++) {{ 
                float acc0 = 0;
                for(int i = 0; i < {start_pos+1}; i++) {{
                    acc0 += temp3[i + {start_pos+1}*(j / 64)] * keys_values[{self.dim*self.max_context} + i*{self.n_heads*64} + j];
                }} 
                xq_temp[j] = acc0;
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min((lidx0+1)*{math.ceil(self.dim/ls)},{self.dim}); i++) {{ 
                float acc = 0;
                for(int x = 0; x < {self.dim}; x++) {{
                    acc += xq_temp[x] * weight[x*{self.dim} + i];
                }}
                h[i] = a[i] + acc + bias[i];
                h_temp[i] = h[i];
            }}
            {barrier[self.d]}
            float total = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min((lidx0+1)*{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                total += h[i];
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean[0] = total / {self.dim};  
            }}
            {barrier[self.d]}
            total = 0;
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                h[i] -= mean[0];
                total += pow(h[i],2);
            }}    
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean[0] = pow(total / {self.dim} + 1e-5,0.5);
            }}
        }}
        """
        if hasattr(self, 'total') == False:
            self.total = transformer.create_buffer_empty(1*4,self.d,self.params)
        if hasattr(self, 'bias3_temp') == False:
            self.bias3_temp = transformer.create_buffer_empty(self.dim*4*4,self.d,self.params)
        if hasattr(self, 'mean') == False:
            self.mean = transformer.create_buffer_empty(1*4,self.d,self.params)
        if hasattr(self, 'h_temp') == False:
            self.h_temp = transformer.create_buffer_empty(self.dim*4,self.d,self.params)
        if hasattr(self, 'h') == False:
            self.h = transformer.create_buffer_empty(self.dim*4,self.d,self.params)
        #transformer.run(prg,"k0_mm",self.params,[a_g,self.mean],1,ls,self.d)
        self.test_ls(k0_mm_prg_str,"k0_mm_"+str(self.dim),[a_g,self.mean],1)
        #transformer.run(prg2,"k0_mm1",self.params,[a_g,c_g,d_g,e_g,xqkv_g,keys_values_g,self.xq_temp_g,self.mean],self.dim*3,ls,self.d)
        #2*MAX_CONTEXT*n_heads*64 is the length is the length of 5
        #(self.n_heads*(self.max_context+1)*64 + 64) length of 6
        self.test_ls(k0_mm1_prg_str,"k0_mm1_"+str(self.dim)+"_"+str(start_pos),[a_g,c_g,d_g,e_g,xqkv_g,keys_values_g,self.xq_temp_g,self.mean],self.dim*3,start_pos)
        #transformer.run(prg2,"k0_mm2",self.params,[keys_values_g ,self.temp_g, self.xq_temp_g],(self.n_heads*(start_pos+1)*(start_pos+1)),ls,self.d)
        self.test_ls(k0_mm2_prg_str,"k0_mm2_"+str(self.dim)+"_"+str(start_pos),[keys_values_g ,self.temp_g, self.xq_temp_g],(self.n_heads*(start_pos+1)*(start_pos+1)),start_pos)

        #transformer.run(prg2,"k0_mm3",self.params,[a_g,keys_values_g,attn_c_proj_weight_g\
        #,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,ls2,self.d)
        #print(self.temp_g.size/4)
        #print(self.max_context*self.n_heads)
        #print(attn_c_proj_weight_g.size/4)
        #exit()
        self.test_ls(k0_mm3_prg_str,"k0_mm3_all"+str(self.dim)+"_"+str(start_pos),[a_g,keys_values_g,attn_c_proj_weight_g,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,start_pos)
        #transformer.run(prg2,"k0_mm3_b",self.params,[a_g,keys_values_g,attn_c_proj_weight_g\
        #,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,ls2,self.d)
        #self.test_ls(k0_mm3_prg_str_b,"k0_mm3_b_"+str(self.dim)+"_"+str(start_pos),[a_g,keys_values_g,attn_c_proj_weight_g,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,start_pos)
        
        #transformer.run(prg2,"k0_mm3_c",self.params,[a_g,keys_values_g,attn_c_proj_weight_g\
        #,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,ls2,self.d)
        #self.test_ls(k0_mm3_prg_str_c,"k0_mm3c_"+str(self.dim)+"_"+str(start_pos),[a_g,keys_values_g,attn_c_proj_weight_g,bias_g,self.temp_g, self.xq_temp_g,self.mean,self.h_temp,self.h],1,start_pos)
        #transformer.run(prg,"k0_mm4",self.params,[a_g,weight2_g,bias2_g,\
        #weight3_g,bias3_g,self.mean,self.h,self.bias3_temp],self.dim*4,ls,self.d)
        self.test_ls(k0_mm4_prg_str,"k0_mm4_"+str(self.dim),[a_g,weight2_g,bias2_g,weight3_g,bias3_g,self.mean,self.h,self.bias3_temp],self.dim*4)
        #transformer.run(prg,"k0_mm5",self.params,[a_g,weight4_g,bias4_g,self.h_temp,self.bias3_temp],self.dim,ls,self.d)
        #weight4 size = 768*768*4
        self.test_ls(k0_mm5_prg_str,"k0_mm5_"+str(self.dim),[a_g,weight4_g,bias4_g,self.h_temp,self.bias3_temp],self.dim)
        return a_g
 
    def kernel_2(self,x_g,ln_1_weight_g,ln_1_bias_g,attn_weight_g,attn_bias_g,cache_kv_g,attn_c_proj_weight_g,attn_c_proj_bias_g,ln_2_weight_g,ln_2_bias_g,c_fc_weight_g,c_fc_bias_g\
        ,c_proj_weight_g,c_proj_bias_g,num_tokens,max_content,j=0):
        if hasattr(self, 'h_g') == False:
            self.h_g = transformer.create_buffer_empty(max_content*self.dim*4,self.d,self.params)
        if hasattr(self, 'h2_g') == False:
            self.h2_g = transformer.create_buffer_empty(max_content*self.dim*4,self.d,self.params)
        if hasattr(self, 'xq_g') == False:
            self.xq_g = transformer.create_buffer_empty(self.n_heads*64*max_content*4,self.d,self.params)
        if hasattr(self, 'xq_g_temp') == False:
            self.xq_g_temp = transformer.create_buffer_empty(self.n_heads*64*max_content*4,self.d,self.params)
        if hasattr(self, 'xv_g') == False:
            self.xv_g = transformer.create_buffer_empty(self.n_heads*64*max_content*4,self.d,self.params)
        if hasattr(self, 'c_g') == False:
            self.c_g = transformer.create_buffer_empty(self.n_heads*64*max_content*4,self.d,self.params)
        if hasattr(self, 'xqt_g') == False:
            self.xqt_g = transformer.create_buffer_empty(self.n_heads*64*max_content*4,self.d,self.params)
        if hasattr(self, 'res_g') == False:
            self.res_g = transformer.create_buffer_empty(max_content*self.n_heads*4,self.d,self.params)
        if hasattr(self, 'xqkv_g') == False:
            self.xqkv_g = transformer.create_buffer_empty(max_content*self.dim*3*4,self.d,self.params)
        if hasattr(self, 'd_g') == False:
            self.d_g = transformer.create_buffer_empty(max_content*self.dim*4*4,self.d,self.params)
        prg_str = f"""
        {kernel_prefix[self.d]}
        {func_dec[self.d]} void k2_mm(
            {var_dec[self.d]} float *x, {var_dec[self.d]} const float *weight, {var_dec[self.d]} const float *bias,
            {var_dec[self.d]} float *copy{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            {local_var[self.d]} float temp2[{num_tokens}];
            int gidx0 = {global_idx[self.d]};
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls};
            temp2[r] = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                copy[i] = x[i];
                temp2[r] += x[i];
            }}
            temp[lidx0] = temp2[r];
            {barrier[self.d]}
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {self.dim};  
            }}
            {barrier[self.d]}
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                x[i] -= temp2[r];
            }}
            {barrier[self.d]}
            temp2[r] = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                temp2[r] += pow(x[i],2);
            }}
            temp[lidx0] = temp2[r];
            {barrier[self.d]}
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {self.dim} + 1e-5,0.5);
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                x[{self.dim}*r + i] = (x[{self.dim}*r + i] * weight[i]) / temp2[r] + bias[i];
            }}
        }}
        {func_dec[self.d]} void k2_mm2(
            {var_dec[self.d]} const float *x, {var_dec[self.d]} const float *attn_weight, {var_dec[self.d]} const float *attn_bias,{var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {self.dim*3*num_tokens}) {{
            int i = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {self.dim}; k++) {{
                total += x[y*{self.dim} + k] * attn_weight[i*{self.dim} + k]; 
            }}
            res[y*{self.dim*3} + i] = total + attn_bias[i];
            }}
        }}
        {func_dec[self.d]} void k2_mm3(
            {var_dec[self.d]} const float *xqkv, {var_dec[self.d]} float *new_cache{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {num_tokens*self.n_heads*64}) {{
                int i = gidx0 / {self.n_heads*64};
                int j = gidx0 % {self.n_heads*64};
                new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*1 + j];
                new_cache[{max_content}*{self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*2 + j]; 
            }}
        }}                 
        {func_dec[self.d]} void k2_tr(
            {var_dec[self.d]} const float *xqkv, {var_dec[self.d]} float *xq, {var_dec[self.d]} float *xv{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {num_tokens*self.n_heads*64}) {{
                int i = (gidx0 / {64}) / {num_tokens};
                int j = (gidx0 / {64}) % {num_tokens};
                int k = gidx0 % 64;
                xq[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k];
                xv[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k + {64*self.n_heads*2}];
            }}
        }}
        {func_dec[self.d]} void k2_ms0(
            {var_dec[self.d]} float *xq_temp, {var_dec[self.d]} const float *xq, {var_dec[self.d]} const float *xqkv{uint3_arg[self.d]})
        {{
                int gidx0 = {global_idx[self.d]};
                if(gidx0 < {self.n_heads*num_tokens*num_tokens}) {{
                int x = (gidx0 / {num_tokens}) % {num_tokens};
                int z = gidx0 / ({num_tokens}*{num_tokens}); 
                int y = gidx0 % {num_tokens};
                float total = 0;
                for(int k = 0; k < {64}; k++) {{
                    //total += xq[y*{64} + k + z*{num_tokens}*{64}] * xqkv[x*{64*self.n_heads*3} + k + z*64 + {self.dim}]; 
                    total += xq[y*{64} + k + z*{num_tokens}*{64}] * xqkv[x*{64*self.n_heads*3} + k + z*64 + {self.dim}]; 
                }}
                xq_temp[y*{num_tokens} + x + z*{num_tokens}*{num_tokens}] = total/8; //sqrt 64 input shape xq TODO min hack
                }}
        }}
        {func_dec[self.d]} void k2_ms(
            {var_dec[self.d]} float *xq{uint3_arg[self.d]})
        {{
        int gidx0 = {global_idx[self.d]};
        if(gidx0 < {self.n_heads*num_tokens*num_tokens}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = gidx0 % {num_tokens};
            if(z > y) {{ //todo, this can probably be 2x faster
                xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = 0;
            }} else {{
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = exp(xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z]);
            }}
        }}
        }}
        {func_dec[self.d]} void k2_ms3(
            {var_dec[self.d]} const float *xq, {var_dec[self.d]} float *mx{uint3_arg[self.d]})
        {{
        int gidx0 = {global_idx[self.d]};
        if(gidx0 < {num_tokens*self.n_heads}) {{
        int x = gidx0 / {num_tokens};
        int y = gidx0 % {num_tokens};
            float m = 0;
            for(int z = 0; z < {num_tokens}; z++) {{
                m += xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z];
            }}
            mx[x*{num_tokens} + y] = m;   //RORY SHOULD BE M
            }}
        }}
        {func_dec[self.d]} void k2_ms4(
            {var_dec[self.d]} float *xq, {var_dec[self.d]} const float *mx{uint3_arg[self.d]})
        {{
        int gidx0 = {global_idx[self.d]};
        if(gidx0 < {self.n_heads*num_tokens*num_tokens}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = (gidx0 % {num_tokens});
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] /= mx[x*{num_tokens} + y];
        }}
        }}
        {func_dec[self.d]} void k2_ms5(
            {var_dec[self.d]} const float *xq, {var_dec[self.d]} const float *xv, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {self.n_heads*num_tokens*64}) {{
                int z = (gidx0 / {num_tokens}) / {64};
                int x = (gidx0 / {num_tokens}) % {64};
                int y = gidx0 % {num_tokens};
                float total = 0;
                for(int k = 0; k < {num_tokens}; k++) {{
                    total += xq[y*{num_tokens} + k + z*{num_tokens}*{num_tokens}] * xv[x + k*{64} + z*{num_tokens}*{64}]; 
                }}
                res[y*{64} + x + z*{64}*{num_tokens}] = total;
            }}
        }}
        {func_dec[self.d]} void k2_ms6( //transpose
            {var_dec[self.d]} const float *xq, {var_dec[self.d]} float *xqt{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {num_tokens*self.n_heads*64}) {{
            int i = (gidx0 / 64) / {num_tokens};
            int j = (gidx0 / 64) % {num_tokens};
            int k = (gidx0 % 64);
            xqt[i*64 + j*{self.n_heads*64} + k] = xq[i*{num_tokens}*64 + j*64 + k];
            }}
        }}
        {func_dec[self.d]} void k2_ms7(
            {var_dec[self.d]} const float *xq, {var_dec[self.d]} const float *attn_weight,{var_dec[self.d]} const float *attn_bias, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {self.dim*num_tokens}) {{
                int x = gidx0 / {num_tokens};
                int y = gidx0 % {num_tokens};
                float total = 0;
                for(int k = 0; k < {self.dim}; k++) {{
                    total += xq[y*{self.dim} + k] * attn_weight[x + k*{self.dim}]; 
                }}
                res[y*{self.dim} + x] += total + attn_bias[x];
            }}
        }}
        {func_dec[self.d]} void k2_ms8( //TODO, there are other kernels like this to fix
            {var_dec[self.d]} float *x, {var_dec[self.d]} const float *ln_2_weight, {var_dec[self.d]} const float *ln_2_bias
            ,{var_dec[self.d]} float *copy{uint3_arg[self.d]})
        {{
            {local_var[self.d]} float temp[{ls}];
            {local_var[self.d]} float total;
            int gidx0 = {global_idx[self.d]};
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; //todo clean
            total = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                copy[i] = x[i];
                total += x[i];
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0<{num_tokens}) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
            }}
            {barrier[self.d]}
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                x[i] -= total / {self.dim};
            }}
            {barrier[self.d]}
            total = 0;
            for(int i = {self.dim}*r + lidx0*{math.ceil(self.dim/ls)}; i < min({self.dim}*r + lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}*r + {self.dim}); i++) {{
                total += pow(x[i],2);
            }}
            temp[lidx0] = total;
            {barrier[self.d]}
            if(lidx0<{num_tokens}) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                total = pow(total / {self.dim} + 1e-5,0.5);
            }}
            {barrier[self.d]}
            for(int i = lidx0*{math.ceil(self.dim/ls)}; i < min(lidx0*{math.ceil(self.dim/ls)}+{math.ceil(self.dim/ls)},{self.dim}); i++) {{
                x[{self.dim}*r + i] = (x[{self.dim}*r + i] * ln_2_weight[i]) / total + ln_2_bias[i];
            }}
        }}
        {func_dec[self.d]} void k2_ms9(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} const float *c_fc_weight,{var_dec[self.d]} const float *c_fc_bias, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            //tanh = ((2.0f*(1/(exp2((val0*(-2.885390043258667f)))+1.0f)))+(-1.0f));
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {self.dim*4*num_tokens}) {{
                int x = gidx0 / {num_tokens};
                int y = gidx0 % {num_tokens};
                float total = 0;
                for(int k = 0; k < {self.dim}; k++) {{
                    total += a[y*{self.dim} + k] * c_fc_weight[x*{self.dim} + k];
                }}
                res[y*{self.dim*4} + x] = 0.5 * (total + c_fc_bias[x])\
                    * (1 + ((2.0f*(1/(exp2(((total + c_fc_bias[x]) * 0.7978845608 * (1 + 0.044715 * pow((total + c_fc_bias[x]),2))*(-2.885390043258667f)))+1.0f)))+(-1.0f)));
            }}
        }}
        {func_dec[self.d]} void k2_ms10(
            {var_dec[self.d]} const float *a, {var_dec[self.d]} const float *c_proj_weight,{var_dec[self.d]} const float *c_proj_bias, {var_dec[self.d]} float *res{uint3_arg[self.d]})
        {{
            int gidx0 = {global_idx[self.d]};
            if(gidx0 < {self.dim*num_tokens}) {{
                int x = gidx0 / {num_tokens};
                int y = gidx0 % {num_tokens};
                float total = 0;
                for(int k = 0; k < {self.dim*4}; k++) {{
                    total += a[y*{self.dim*4} + k] * c_proj_weight[x*{self.dim*4} + k];
                }}
                res[y*{self.dim} + x] += total + c_proj_bias[x];
            }}
        }}
        """
        if prg_str not in self.prg_cache:
            library = transformer.compile(prg_str,self.d,self.params)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]
        transformer.run(prg,"k2_mm",self.params,[x_g,ln_1_weight_g,ln_1_bias_g,self.h_g],num_tokens*ls,ls,self.d)
        transformer.run(prg,"k2_mm2",self.params,[x_g,attn_weight_g,attn_bias_g,self.xqkv_g],self.dim*3*num_tokens,ls,self.d)
        transformer.run(prg,"k2_mm3",self.params,[self.xqkv_g, cache_kv_g],num_tokens*self.n_heads*64,ls,self.d)
        transformer.run(prg,"k2_tr",self.params,[self.xqkv_g, self.xq_g, self.xv_g],num_tokens*self.n_heads*64,ls,self.d)
        transformer.run(prg,"k2_ms0",self.params,[self.xq_g_temp,self.xq_g, self.xqkv_g],self.n_heads*num_tokens*num_tokens,ls,self.d) #TODO check again
        transformer.run(prg,"k2_ms",self.params,[self.xq_g_temp],self.n_heads*num_tokens*num_tokens,ls,self.d)
        transformer.run(prg,"k2_ms3",self.params,[self.xq_g_temp,self.res_g],self.n_heads*num_tokens,ls,self.d)
        transformer.run(prg,"k2_ms4",self.params,[self.xq_g_temp,self.res_g],self.n_heads*num_tokens*num_tokens,ls,self.d)
        transformer.run(prg,"k2_ms5",self.params,[self.xq_g_temp,self.xv_g,self.c_g],num_tokens*self.n_heads*64,ls,self.d)
        transformer.run(prg,"k2_ms6",self.params,[self.c_g,self.xqt_g],num_tokens*self.n_heads*64,ls,self.d)
        transformer.run(prg,"k2_ms7",self.params,[self.xqt_g,attn_c_proj_weight_g,attn_c_proj_bias_g,self.h_g],self.dim*num_tokens,ls,self.d)
        transformer.run(prg,"k2_ms8",self.params,[self.h_g, ln_2_weight_g, ln_2_bias_g,self.h2_g],num_tokens*ls,ls,self.d)
        transformer.run(prg,"k2_ms9",self.params,[self.h_g, c_fc_weight_g,c_fc_bias_g,self.d_g],self.dim*4*num_tokens,ls,self.d)
        transformer.run(prg,"k2_ms10",self.params,[self.d_g, c_proj_weight_g,c_proj_bias_g,self.h2_g],self.dim*num_tokens,ls,self.d)
        return self.h2_g

    def time_it(func,a,b,i=100):
        f = None
        total_time = 0
        for _ in range(i):
            st = time.perf_counter()
            ret = func(a,b)
            t = time.perf_counter() - st
            total_time += t
            if f is None or t < f:
                f = t
        return ret,f

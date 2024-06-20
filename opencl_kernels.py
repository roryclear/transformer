import numpy as np
import pyopencl as cl
import time
import math

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
dim = 768
prg = None
n_heads = 12

class Opencl_Kernels:
    def __init__(self):
        self.prg_cache = {}
        self.buffer_cache = {}
        return None

    def add(self,a_g,b_g,b_s=0,a_s=0):
        if "add_res" not in self.buffer_cache:
            self.buffer_cache["add_res"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(768).astype(np.float32))
        res_g = self.buffer_cache["add_res"]
        prg_str = f"""
        __kernel void add(
            __global const float *a, __global const float *b, __global float *res)
        {{
        int gidx0 = get_global_id(0);
            res[gidx0] = a[{int(a_s)*768} + gidx0] + b[gidx0 + {b_s}*768];   
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]
        knl = prg.add
        knl(queue, (768,1), (256,1), a_g, b_g,res_g) #todo check shape
        return res_g

    len = 768
    loop_size = int(len / 256)
    len_short = 768

    def tok_emb(self,tokens,weight_g,weight_2_g,no_tokens):
        tokens_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tokens)
        ls = 256
        size = no_tokens*dim
        tok_emb = np.zeros((no_tokens,dim)).astype(np.float32)
        if "tok_emb" not in self.buffer_cache:
            self.buffer_cache["tok_emb"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros((no_tokens,dim)).astype(np.float32))
        tok_emb_g = self.buffer_cache["tok_emb"]
        prg_str = f"""
        __kernel void mm(
            __global int *tokens, __global const float *weight, __global const float *weight2,  __global float *tok_emb)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {dim};
            int j = gidx0 % {dim};
            tok_emb[i*{dim} + j] = weight[tokens[i]*{dim} + j] + weight2[i*{dim} + j];
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx, prg_str).build()
        prg = self.prg_cache[prg_str]
        knl = prg.mm
        knl(queue, (math.ceil(size / ls)*ls,1), (ls,1), tokens_g, weight_g, weight_2_g,tok_emb_g)
        return tok_emb_g

    def kernel_0_b(self,x_g,weight_g,bias_g,attn_weight_g,attn_bias_g\
        ,ln_f_weight_g,ln_f_bias_g,lm_head_weight_g,new_cache_g,temperature,n_tokens,random_num,max_content,retnp=False):
        size = 768 #todo hardcoded
        ls = 256
        b_cols = 2304 #todo
        b_rows = 768
        b_cols2 = 50257
        seg = int(size / ls) #todo
        x0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(n_tokens*768).astype(np.float32))
        seg2 = math.ceil(50257 / ls)
        prg_str = f"""
        __kernel void mm0(
            __global float *x, __global const float *ln_f_weight, __global const float *ln_f_bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{seg}];
            __attribute__ ((aligned (16))) __local float mean;
            int lidx0 = get_local_id(0);
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += x[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {size};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[i + lidx0*{seg} + {(n_tokens - 1)*768}] -= mean;
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(x[lidx0*{seg} + i + {(n_tokens - 1)*768}],2);
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {size} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[i + lidx0*{seg}] = (x[i + lidx0*{seg} + {(n_tokens - 1)*768}] * ln_f_weight[i + lidx0*{seg}]) / mean + ln_f_bias[i + lidx0*{seg}];
            }}
        }}

        __kernel void mm(
            __global float *x, __global const float *x2, __global const float *weight, __global const float *bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{n_tokens}];
            int lidx0 = get_local_id(0);
            int gidx0 = get_group_id(0);
            int r = gidx0; //todo clean
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + lidx0*{seg} + i] = x2[768*r + lidx0*{seg} + i];
                temp2[r] += x[768*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {size};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[768*r + lidx0*{seg} + i],2);
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {size} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + i + lidx0*{seg}] = (x[768*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
            }}
        }}

        __kernel void mm2(
            __global const float *x, __global const float *attn_weight, __global const float *attn_bias,__global float *res)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {n_tokens};
            int y = gidx0 % {n_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += x[y*{b_rows} + k] * attn_weight[i*{b_rows} + k]; 
            }}
            res[y*{b_cols} + i] = total + attn_bias[i];
        }}

        __kernel void mm3(
            __global const float *xqkv, __global float *new_cache)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {12*64};
            int j = gidx0 % {12*64};
            new_cache[i*12*64 + j] = xqkv[i*12*64*3 + 768*1 + j];
            new_cache[{max_content}*12*64 + i*12*64 + j] = xqkv[i*12*64*3 + 768*2 + j]; 
        }}

        __kernel void mm4(
            __global const float *xqkv, __global float *new_cache)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {12*64};
            int j = gidx0 % {12*64};
            new_cache[i*12*64 + j] = xqkv[i*12*64*3 + j + 768];
            new_cache[{max_content}*12*64 + i*12*64 + j] = xqkv[i*12*64*3 + j + 768*2]; 
        }}

        __kernel void mm5(
            __global const float *a, __global const float *lm_head_weight, __global float *res)
        {{
            int x = get_global_id(0);
            if(x < {b_cols2}) {{
                float total = 0;
                for(int k = 0; k < {b_rows}; k++) {{
                    total += a[k] * lm_head_weight[x*{b_rows} + k]; 
                }}
                res[x] = total / {temperature}; 
            }}
        }}

                __kernel void mm6(
            __global const float *a, __global float *res)
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        __kernel void mm7(
        __global float *a, __global const float *res)
        {{
            barrier(CLK_LOCAL_MEM_FENCE);
            int gidx0 = get_global_id(0);
            a[gidx0] = exp(a[gidx0] - res[0]);
        }}

        __kernel void mm8(
            __global const float *a, __global float *res)
        {{
            res[0] = a[0];
        }}

        __kernel void mm9(
        __global float *a, __global const float *res)
        {{
            int gidx0 = get_global_id(0);
            a[gidx0] = a[gidx0] / res[0];
        }}

        __kernel void mm10(
        __global const float *a, __global float *res)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            int lidx0 = get_local_id(0);
            float t = 0;
            for(int i = 0; i < {seg2}; i++) {{
                t += a[lidx0*{seg2} + i];
            }}
            temp[lidx0] = t;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0 == 0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                res[0] = t;
            }}
        }}

        __kernel void mm11(
        __global float *a)
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}

        __kernel void mm12(
        __global float *a)
        {{
            int gidx0 = get_global_id(0);
            if((a[gidx0] / a[50256]) < {random_num}) {{
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}

        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx, prg_str).build()
        prg = self.prg_cache[prg_str]
        knl = prg.mm
        knl(queue, (ls*n_tokens,1), (ls,1), x0_g, x_g, weight_g, bias_g) 
        g = math.ceil((b_cols*n_tokens / ls)*ls)
        if "c" not in self.buffer_cache:
            self.buffer_cache["c"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(n_tokens*b_cols).astype(np.float32))
        c_g = self.buffer_cache["c"]
        knl2 = prg.mm2
        knl2(queue, (g,1), (ls,1), x0_g, attn_weight_g,attn_bias_g,c_g)

        knl4 = prg.mm4
        ls = 256
        g = math.ceil((n_tokens*12*64) / ls) * ls
        knl4(queue, (g,1), (ls,1), c_g, new_cache_g) 

        knl = prg.mm0
        knl(queue, (ls,1), (ls,1), x_g, ln_f_weight_g, ln_f_bias_g)

        if "c2" not in self.buffer_cache:
            self.buffer_cache["c2"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(b_cols2).astype(np.float32))
        c_g = self.buffer_cache["c2"]

        knl = prg.mm5
        group_size = math.ceil(b_cols2 / ls) * ls
        knl(queue, (group_size,1), (ls,1), x_g, lm_head_weight_g,c_g)

        res = np.zeros(1).astype(np.float32)
        if "res" not in self.buffer_cache:
            self.buffer_cache["res"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(1).astype(np.float32))
        res_g = self.buffer_cache["res"]

        knl = prg.mm6
        knl(queue, (1,1), (1,1), c_g, res_g) 
        knl2 = prg.mm7
        knl2(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), c_g, res_g)
        knl3 = prg.mm8
        knl3(queue, (1,1), (1,1), c_g, res_g)
        knl4 = prg.mm9
        knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), c_g, res_g)
        knl5 = prg.mm10
        knl5(queue, (ls,1), (ls,1), c_g, res_g)
        knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), c_g, res_g)
        knl6 = prg.mm11
        knl6(queue, (1,1), (1,1), c_g)
        knl7 = prg.mm12
        knl7(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), c_g)
        knl5(queue, (ls,1), (ls,1), c_g, res_g)
        cl.enqueue_copy(queue, res, res_g)
        return res
    
    def kernel_2(self,a_g,c_g,d_g,e_g,xqkv_g,g,keys_values_g,start_pos,weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,max_content): #g = size
        ls = 256
        seg = int(dim / ls) #todo
        seg3 = math.ceil(12*(start_pos+1)*(start_pos+1) / ls)
        if "temp" not in self.buffer_cache:
            self.buffer_cache["temp"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(12*(max_content)).astype(np.float32))
        temp_g = self.buffer_cache["temp"]
        if "xq_temp" not in self.buffer_cache:
            self.buffer_cache["xq_temp"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(768).astype(np.float32))
        xq_temp_g = self.buffer_cache["xq_temp"]
        prg_str = f"""
        __kernel void mm(
            __global float *a, __global const float *c, __global const float *d, __global const float *e,
            __global const float *xqkv, __global float *keys_values,
            __global float *xq_temp)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float mean;
            int lidx0 = get_local_id(0);
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += a[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {dim};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                a[i + lidx0*{seg}] -= mean;
                total += pow(a[lidx0*{seg} + i],2);
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {dim} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {int(dim*3 / ls)}; i++) {{
                xq_temp[lidx0*{int(dim*3 / ls)} + i] = xqkv[lidx0*{int(dim*3 / ls)} + i];
                float total = 0;
                for(int k = 0; k < {dim}; k++) {{
                    total += ((a[k] * c[k]) / mean + d[k]) * e[(lidx0*{int(dim*3 / ls)} + i)*{dim} + k];
                }}
                if((lidx0*{int(dim*3 / ls)} + i) < {g}) {{
                    xq_temp[lidx0*{int(dim*3 / ls)} + i] += total;
                    }}
                if((lidx0*{int(dim*3 / ls)} + i) >= {g} && (lidx0*{int(dim*3 / ls)} + i) < {2*g}) {{
                    keys_values[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {g}] = xqkv[768 + lidx0*{int(dim*3 / ls)} + i - {g}] + total;
                }}
                if((lidx0*{int(dim*3 / ls)} + i) >= {2*g}) {{
                    keys_values[98304 + {start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {2*g}] = xqkv[{768*2} + lidx0*{int(dim*3 / ls)} + i - {2*g}] + total;
                }}
            }}
        }}

        __kernel void mm2(
            __global float *keys_values,
            __global float *temp3, __global float *xq_temp)
        {{
            int lidx0 = get_global_id(0);
                int x = (lidx0) % {start_pos+1};
                int k = (lidx0) / {start_pos+1};
                float acc0 = 0;
                for(int i = 0; i < 64; i++) {{
                    acc0 += xq_temp[i + 64*k] * keys_values[x*12*64 + i + 64*k];
                }}                  
                temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
            __kernel void mm3(
            __global float *a,
            __global float *keys_values,
            __global const float *weight,__global const float *bias,
            __global const float *weight2, __global const float *bias2,
            __global const float *weight3, __global const float *bias3,
            __global const float *weight4,
            __global float *bias4,
            __global float *temp3, __global float *xq_temp)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float mean;
            __attribute__ ((aligned (16))) __local float bias3_temp[3072];
            __attribute__ ((aligned (16))) __local float bias4_temp[2304];
            __attribute__ ((aligned (16))) __local float h_temp[768];
            __attribute__ ((aligned (16))) __local float h[768];
            int lidx0 = get_local_id(0);
            if(lidx0 < 12){{
            float m = -INFINITY;
            for(int i = 0; i < {start_pos+1}; i++) {{
                float val = temp3[i + lidx0*{start_pos+1}];
                m = max(m,val);
            }}
            float t = 0;
            for(int i = 0; i < {start_pos+1}; i++) {{
                temp3[i + lidx0*{start_pos+1}] = exp(temp3[i + lidx0*{start_pos+1}] - m);
                float val = temp3[i + lidx0*{start_pos+1}];
                t = t+val;
            }}
            for(int i = 0; i < {start_pos+1}; i++) {{
                temp3[i + lidx0*{start_pos+1}] = temp3[i + lidx0*{start_pos+1}] / t;
            }}
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int g = 0; g < {seg}; g++) {{
                int y = (g + lidx0*{seg}) / 64;
                int x = (g + lidx0*{seg}) % 64;
                float acc0 = 0;
                for(int i = 0; i < {start_pos+1}; i++) {{
                    acc0 += temp3[i + {start_pos+1}*y] * keys_values[98304 + i*12*64 + x + y*64];
                }}
                xq_temp[x + y*64] = acc0;
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                float acc = 0;
                for(int x = 0; x < {dim}; x++) {{
                    acc += xq_temp[x] * weight[x*{dim} + lidx0*{seg} + i];
                }}
                h[lidx0*{seg} + i] = a[lidx0*{seg} + i] + acc + bias[lidx0*{seg} + i];
                h_temp[lidx0*{seg} + i] = h[lidx0*{seg} + i];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += h[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {dim};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] = h[i + lidx0*{seg}] - mean;
                total += pow(h[lidx0*{seg} + i],2);
            }}        
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {dim} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {int(dim*4 / ls)}; i++) {{
                bias3_temp[i + lidx0*{int(dim*4 / ls)}] = bias3[i + lidx0*{int(dim*4 / ls)}];
                for(int j = 0; j < {dim}; j++) {{
                    bias3_temp[i + lidx0*{int(dim*4 / ls)}] += ((h[j] * weight2[j]) / mean + bias2[j]) * weight3[(i + lidx0*{int(dim*4 / ls)})*{dim} + j];
                }}
                bias3_temp[i + lidx0*{int(dim*4 / ls)}] = 0.5 * bias3_temp[i + lidx0*{int(dim*4 / ls)}]\
                * (1 + tanh(bias3_temp[i + lidx0*{int(dim*4 / ls)}] * 0.7978845608\
                * (1 + 0.044715 * pow(bias3_temp[i + lidx0*{int(dim*4 / ls)}],2))));
            }}
            barrier(CLK_LOCAL_MEM_FENCE);  
            for(int i = 0; i < {int(dim / ls)}; i++) {{
                bias4_temp[lidx0 + i*{ls}] = bias4[lidx0 + i*{ls}];
                for(int j = 0; j < {dim*4}; j++) {{
                    bias4_temp[lidx0 + i*{ls}] += bias3_temp[j] * weight4[lidx0 + i*{ls} + j*{dim}];
                }}
                a[lidx0 + i*{ls}] = bias4_temp[lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
            }}
        }}
        
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]

        knl = prg.mm
        knl2 = prg.mm2
        knl3 = prg.mm3
        knl(queue, (ls,1), (ls,1),a_g,c_g,d_g,e_g,xqkv_g\
        ,keys_values_g,xq_temp_g)
        knl2(queue, (ls*seg3,1), (ls,1),keys_values_g,temp_g, xq_temp_g)
        knl3(queue, (ls,1), (ls,1),a_g\
        ,keys_values_g,weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,temp_g, xq_temp_g)
        return a_g

    def kernel_8(self,h_g,lm_head_weight_g,ln_f_weight_g,ln_f_bias_g,random_num,temperatue):
        rows = 768
        cols = 50257
        if "logits" not in self.buffer_cache:
            self.buffer_cache["logits"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(cols).astype(np.float32))
        logits_g = self.buffer_cache["logits"]
        ls = 256
        seg = math.ceil(50257 / ls)
        res = np.zeros(1).astype(np.float32)
        if "res" not in self.buffer_cache:
            self.buffer_cache["res"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
        res_g = self.buffer_cache["res"]
        prg_str = f"""

        __kernel void mm0(
            __global float *h, __global const float *weight, __global const float *bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{seg}];
            __attribute__ ((aligned (16))) __local float mean;
            int lidx0 = get_local_id(0);
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += h[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {dim};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] -= mean;
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(h[lidx0*{seg} + i],2);
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {dim} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] = (h[i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / mean + bias[i + lidx0*{seg}];
            }}
        }}

        __kernel void matvec(
            __global const float *h, __global const float *weight2 , __global float *logits)
        {{
            int gidx0 = get_global_id(0);
            logits[gidx0] = 0;
            for(int j = 0; j < {rows}; j++) {{
                logits[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
            }}
            logits[gidx0] /= {temperatue};
        }}

        __kernel void mm(
            __global const float *a, __global float *logits)
        {{
            logits[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        __kernel void mm2(
        __global float *a, __global const float *logits)
        {{
            barrier(CLK_LOCAL_MEM_FENCE);
            int gidx0 = get_global_id(0);
            a[gidx0] = exp(a[gidx0] - logits[0]);
        }}

        __kernel void mm3(
            __global const float *a, __global float *logits)
        {{
            logits[0] = a[0];
        }}

        __kernel void mm4(
        __global float *a, __global const float *logits)
        {{
            int gidx0 = get_global_id(0);
            a[gidx0] = a[gidx0] / logits[0];
        }}

        __kernel void mm5(
        __global const float *a, __global float *logits)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            int lidx0 = get_local_id(0);
            float t = 0;
            for(int i = 0; i < {seg}; i++) {{
                t += a[lidx0*{seg} + i];
            }}
            temp[lidx0] = t;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0 == 0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                logits[0] = t;
            }}
        }}

        __kernel void mm6(
        __global float *a)
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}

        __kernel void mm7(
        __global float *a)
        {{
            int gidx0 = get_global_id(0);
            if((a[gidx0] / a[50256]) < {random_num}) {{
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]

        knl = prg.mm0
        knl(queue, (ls,1), (ls,1), h_g, ln_f_weight_g, ln_f_bias_g) 

        knl = prg.matvec
        gidx = math.ceil(cols / 16) * 16
        knl(queue, (gidx,1), (16,1), h_g, lm_head_weight_g, logits_g)
        
        knl = prg.mm
        knl(queue, (1,1), (1,1), logits_g, res_g) 
        knl2 = prg.mm2
        knl2(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g, res_g)
        knl3 = prg.mm3
        knl3(queue, (1,1), (1,1), logits_g, res_g)
        knl4 = prg.mm4
        knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g, res_g)
        knl5 = prg.mm5
        knl5(queue, (ls,1), (ls,1), logits_g, res_g)
        knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g, res_g)
        knl6 = prg.mm6
        knl6(queue, (1,1), (1,1), logits_g)
        knl7 = prg.mm7
        knl7(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g)
        knl5(queue, (ls,1), (ls,1), logits_g, res_g)
        cl.enqueue_copy(queue, res, res_g)
        return res

    def kernel_7(self,x_g,ln_1_weight_g,ln_1_bias_g,attn_weight_g,attn_bias_g,cache_kv_g,attn_c_proj_weight_g,attn_c_proj_bias_g,ln_2_weight_g,ln_2_bias_g,c_fc_weight,c_fc_bias_g\
        ,c_proj_weight_g,c_proj_bias_g,num_tokens,max_content):
        h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_tokens*768).astype(np.float32))
        h_g_2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_tokens*768).astype(np.float32))
        xq = np.zeros(12*64*num_tokens).astype(np.float32) #todo
        xv = np.zeros(12*64*num_tokens).astype(np.float32) #todo
        xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
        xv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xv)
        c_fc_weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_fc_weight)

        a_rows = num_tokens
        a_cols = 64
        b_rows = 768
        n = 12
        x = 12
        res = np.zeros(num_tokens*x).astype(np.float32)
        res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
        ls = 256
        size = 768
        seg = int(size / ls) #todo
        b_cols = 2304 # for first part
        b_cols_2 = 3072
        prg_str = f"""
        __kernel void mm(
            __global float *x, __global const float *weight, __global const float *bias, __global float *x2, __global float *x3)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{num_tokens}];
            int lidx0 = get_local_id(0);
            int gidx0 = get_group_id(0);
            int r = gidx0; //todo clean
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += x[768*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {size};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x2[768*r + i + lidx0*{seg}] = x[768*r + i + lidx0*{seg}];
                x3[768*r + i + lidx0*{seg}] = x[768*r + i + lidx0*{seg}]; 
                x[768*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[768*r + lidx0*{seg} + i],2);
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {size} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + i + lidx0*{seg}] = (x[768*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
            }}
        }}

        __kernel void mm2(
            __global const float *x, __global const float *attn_weight, __global const float *attn_bias,__global float *res)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += x[y*{b_rows} + k] * attn_weight[i*{b_rows} + k]; 
            }}
            res[y*{b_cols} + i] = total + attn_bias[i];
        }}

        __kernel void mm3(
            __global const float *xqkv, __global float *new_cache)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {12*64};
            int j = gidx0 % {12*64};
            new_cache[i*12*64 + j] = xqkv[i*12*64*3 + 768*1 + j];
            new_cache[{max_content}*12*64 + i*12*64 + j] = xqkv[i*12*64*3 + 768*2 + j]; 
        }}
              
        __kernel void tr(
            __global const float *xqkv, __global float *xq, __global float *xv)
        {{
            int gidx0 = get_global_id(0);
            int i = (gidx0 / {64}) / {num_tokens};
            int j = (gidx0 / {64}) % {num_tokens};
            int k = gidx0 % 64;
            xq[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*64*12*3 + k];
            xv[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*64*12*3 + k + 64*12*2];
        }}

        __kernel void ms0(
            __global float *xq, __global const float *xqkv)
        {{
            int gidx0 = get_global_id(0);
            if(gidx0 < {n}*{a_rows}*{a_rows}) {{
                int x = (gidx0 / {a_rows}) % {a_rows};
                int z = gidx0 / ({a_rows}*{a_rows}); 
                int y = gidx0 % {a_rows};
                float total = 0;
                for(int k = 0; k < {a_cols}; k++) {{
                    total += xq[y*{a_cols} + k + z*{a_rows}*{a_cols}] * xqkv[x*64*12*3 + k + z*64 + 768]; 
                }}
                xq[y*{a_rows} + x + z*{a_rows}*{a_rows}] = total / 8; //sqrt 64 input shape xq
            }}
        }}

        __kernel void ms(
            __global float *xq)
        {{
        int gidx0 = get_global_id(0);
        if(gidx0 < {num_tokens*num_tokens*x}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = gidx0 % {num_tokens};
            if(z > y) {{ //todo, this can probably be 2x faster
                xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = -INFINITY;
            }}
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = exp(xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z]);
        }}
        }}

        __kernel void ms3(
            __global float *xq, __global float *mx)
        {{
        int gidx0 = get_global_id(0);
        int x = gidx0 / {num_tokens};
        int y = gidx0 % {num_tokens};
            float m = 0;
            for(int z = 0; z < {num_tokens}; z++) {{
                m += xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z];
            }}
            mx[x*{num_tokens} + y] = m;  
        }}

        __kernel void ms4(
            __global float *xq, global const float *mx)
        {{
        int gidx0 = get_global_id(0);
        if(gidx0 < {num_tokens*num_tokens*x}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = gidx0 % {num_tokens};
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] /= mx[x*{num_tokens} + y];
        }}
        }}

        __kernel void ms5(
            __global const float *xq, __global const float *xv, __global float *res)
        {{
            int gidx0 = get_global_id(0);
            int z = (gidx0 / {num_tokens}) / {a_cols};
            int x = (gidx0 / {num_tokens}) % {a_cols};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {num_tokens}; k++) {{
                total += xq[y*{num_tokens} + k + z*{num_tokens}*{num_tokens}] * xv[x + k*{a_cols} + z*{num_tokens}*{a_cols}]; 
            }}
            res[y*{a_cols} + x + z*{a_cols}*{num_tokens}] = total;
        }}

        __kernel void ms6( //transpose
            __global const float *xq, __global float *xqt)
        {{
            int gidx0 = get_global_id(0);
            int i = (gidx0 / 64) / {num_tokens};
            int j = (gidx0 / 64) % {num_tokens};
            int k = gidx0 % 64;
            xqt[i*64 + j*12*64 + k] = xq[i*{num_tokens}*64 + j*64 + k];
        }}

        __kernel void ms7(
            __global const float *xq, __global const float *attn_weight,__global const float *attn_bias, __global float *res,
            __global float *res_temp)
        {{
            int gidx0 = get_global_id(0);
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += xq[y*{b_rows} + k] * attn_weight[x*{b_rows} + k]; 
            }}
            res[y*{b_rows} + x] += total + attn_bias[x];
            res_temp[y*{b_rows} + x] = res[y*{b_rows} + x];
        }}

        __kernel void ms8(
            __global float *x, __global const float *ln_2_weight, __global const float *ln_2_bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{num_tokens}];
            int lidx0 = get_local_id(0);
            int gidx0 = get_group_id(0);
            int r = gidx0; //todo clean
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += x[768*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {size};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[768*r + lidx0*{seg} + i],2);
            }}
            temp[lidx0] = temp2[r];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {size} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                x[768*r + i + lidx0*{seg}] = (x[768*r + i + lidx0*{seg}] * ln_2_weight[i + lidx0*{seg}]) / temp2[r] + ln_2_bias[i + lidx0*{seg}];
            }}
        }}

        __kernel void ms9(
            __global const float *a, __global const float *c_fc_weight,__global const float *c_fc_bias, __global float *res)
        {{
            int gidx0 = get_global_id(0);
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += a[y*{b_rows} + k] * c_fc_weight[x*{b_rows} + k]; 
            }}
            res[y*{b_cols_2} + x] = 0.5 * (total + c_fc_bias[x])\
                * (1 + tanh((total + c_fc_bias[x]) * 0.7978845608\
                * (1 + 0.044715 * pow((total + c_fc_bias[x]),2))));
        }}

        __kernel void ms10(
            __global const float *a, __global const float *c_proj_weight,__global const float *c_proj_bias, __global float *res)
        {{
            int gidx0 = get_global_id(0);
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_cols_2}; k++) {{
                total += a[y*{b_cols_2} + k] * c_proj_weight[x*{b_cols_2} + k]; 
            }}
            res[y*{b_rows} + x] += total + c_proj_bias[x];
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]
        knl = prg.mm
        knl(queue, (ls*num_tokens,1), (ls,1), x_g, ln_1_weight_g, ln_1_bias_g, h_g, h_g_2) 
        g = math.ceil((b_cols*num_tokens / ls)*ls)
        xqkv = np.zeros([num_tokens,b_cols]).astype(np.float32)
        xqkv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xqkv)
        knl2 = prg.mm2
        knl2(queue, (g,1), (ls,1), x_g, attn_weight_g,attn_bias_g,xqkv_g)
        g = math.ceil((num_tokens*12*64) / ls) * ls
        knl3 = prg.mm3
        knl3(queue, (g,1), (ls,1), xqkv_g, cache_kv_g)

        knltr = prg.tr
        g = num_tokens*12*64
        ls = 256
        g = math.ceil(g / ls)*ls
        knltr(queue, (g,1), (ls,1), xqkv_g, xq_g, xv_g)

        g = x*num_tokens*num_tokens
        g = math.ceil(g / ls) * ls
        knl0 = prg.ms0
        knl0(queue, (g,1), (ls,1), xq_g, xqkv_g)
        knl = prg.ms
        knl(queue, (g,1), (ls,1), xq_g)
        g2 = x*num_tokens #todo, will break for larger inputs
        knl3 = prg.ms3
        knl3(queue, (g2,1), (g2,1), xq_g,res_g)
        knl4 = prg.ms4
        knl4(queue, (g,1), (ls,1), xq_g,res_g)

        c = np.zeros([12,num_tokens,a_cols])
        c = np.float32(c)
        c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
        g3 = (math.ceil(12*a_cols*num_tokens / ls) * ls)
        knl5 = prg.ms5
        knl5(queue, (g3,1), (ls,1), xq_g,xv_g,c_g)
        g4 = num_tokens*12*64
        g4 = math.ceil(g4 / ls)*ls
        xqt = np.zeros(12*64*num_tokens).astype(np.float32) #todo
        xqt_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xqt)
        knl6 = prg.ms6
        knl6(queue, (g4,1), (ls,1), c_g,xqt_g)

        g = math.ceil((b_rows*num_tokens / ls)*ls)
        knl7 = prg.ms7
        knl7(queue, (g,1), (ls,1), xqt_g,attn_c_proj_weight_g,attn_c_proj_bias_g,h_g,h_g_2)

        knl8 = prg.ms8
        knl8(queue, (ls*num_tokens,1), (ls,1), h_g, ln_2_weight_g, ln_2_bias_g)

        d = np.zeros([num_tokens,b_cols_2]).astype(np.float32)
        d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
        g = math.ceil((b_cols_2*num_tokens / ls)*ls)
        knl9 = prg.ms9
        knl9(queue, (g,1), (ls,1), h_g, c_fc_weight_g,c_fc_bias_g,d_g)

        knl10 = prg.ms10
        g = math.ceil((b_rows*num_tokens / ls)*ls)
        knl10(queue, (g,1), (ls,1), d_g, c_proj_weight_g,c_proj_bias_g,h_g_2)
        #h = np.zeros(13*768).astype(np.float32)
        #cl.enqueue_copy(queue, h, h_g_2)
        return h_g_2
    
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
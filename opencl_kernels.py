import numpy as np
import pyopencl as cl
import time
import math
import transformer

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = None

params = {"ctx":ctx,"mf":mf}

def create_cl_buffer(a):
  return transformer.create_buffer(a,"OpenCL",params)

def create_cl_buffer_empty(size):
  return transformer.create_buffer_empty(size,"OpenCL",params)

class Opencl_Kernels:
    def __init__(self,dim,n_heads,max_context):
        self.prg_cache = {}
        self.dim = dim
        self.n_heads = n_heads
        self.max_context = max_context

    def add(self,a_g,b_g,b_s=0,a_s=0):
        if hasattr(self, 'add_res_g') == False:
            self.add_res_g = create_cl_buffer_empty(self.dim*4)
        prg_str = f"""
        __kernel void add(
            __global const float *a, __global const float *b, __global float *res)
        {{
        int gidx0 = get_global_id(0);
            res[gidx0] = a[{int(a_s)*self.dim} + gidx0] + b[gidx0 + {b_s*self.dim}];   
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]
        prg.add(queue, (self.dim,1), (256,1), a_g.data, b_g.data,self.add_res_g.data) #todo check shape
        return self.add_res_g

    def tok_emb(self,tokens,weight_g,weight_2_g,no_tokens):
        tokens_g = create_cl_buffer(tokens)
        ls = 256
        size = no_tokens*self.dim
        tok_emb_g = create_cl_buffer_empty(no_tokens*self.dim*4)
        prg_str = f"""
        __kernel void mm(
            __global int *tokens, __global const float *weight, __global const float *weight2,  __global float *tok_emb)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {self.dim};
            int j = gidx0 % {self.dim};
            tok_emb[i*{self.dim} + j] = weight[tokens[i]*{self.dim} + j] + weight2[i*{self.dim} + j];
        }}
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]
        prg.mm(queue, (math.ceil(size / ls)*ls,1), (ls,1), tokens_g.data, weight_g.data, weight_2_g.data,tok_emb_g.data)
        return tok_emb_g

    def kernel_1(self,h_g,weight_g,bias_g,weight2_g,temperature,random_num):
        ls = 256
        seg = int(self.dim / ls) #todo
        rows = self.dim
        cols = 50257
        if hasattr(self, 'logits_g') == False:
            self.logits_g = create_cl_buffer_empty(50257*4)
        if hasattr(self, 'res') == False:
            self.res = np.zeros(1).astype(np.float32)
        if hasattr(self, 'res_g') == False:
            self.res_g = create_cl_buffer_empty(1*4)
        seg2 = math.ceil(50257 / ls)
        prg_str = f"""
        __kernel void mm4(
            __global float *h, __global const float *weight, __global const float *bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float mean;
            int lidx0 = get_global_id(0);
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
                mean = total / {self.dim};  
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
                mean = pow(total / {self.dim} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] = (h[i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / mean + bias[i + lidx0*{seg}];
            }}
        }}
        __kernel void matvec(
            __global const float *h, __global const float *weight2 , __global float *res)
        {{
            int gidx0 = get_global_id(0);
            res[gidx0] = 0;
            for(int j = 0; j < {rows}; j++) {{
                res[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
            }}
            res[gidx0] /= {temperature};
        }}
        __kernel void mm5(
            __global const float *a, __global float *res)
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        __kernel void mm6(
        __global float *a, __global const float *res)
        {{
            int gidx0 = get_global_id(0);
            a[gidx0] = exp(a[gidx0] - res[0]);
        }}

        __kernel void mm7(
            __global const float *a, __global float *res)
        {{
            res[0] = a[0];
        }}

        __kernel void mm8(
        __global float *a, __global const float *res)
        {{
            int gidx0 = get_global_id(0);
            a[gidx0] = a[gidx0] / res[0];
        }}

        __kernel void mm9(
        __global const float *a, __global float *res)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            int lidx0 = get_global_id(0);
            float t = 0;
            for(int i = 0; i < {math.ceil(50257 / ls)}; i++) {{
                t += a[lidx0*{math.ceil(50257 / ls)} + i];
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

        __kernel void mm10(
        __global float *a)
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}

        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]

        #different every time
        prg_str = f""" 
        __kernel void mm11(
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
        prg2 = cl.Program(ctx,prg_str).build()

        prg.mm4(queue, (ls,1), (ls,1), h_g.data, weight_g.data, bias_g.data)
        gidx = math.ceil(cols / 16) * 16
        prg.matvec(queue, (gidx,1), (16,1), h_g.data, weight2_g.data,self.logits_g.data)

        prg.mm5(queue, (1,1), (1,1), self.logits_g.data, self.res_g.data) 
        prg.mm6(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), self.logits_g.data, self.res_g.data)
        prg.mm7(queue, (1,1), (1,1), self.logits_g.data, self.res_g.data)
        prg.mm8(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), self.logits_g.data, self.res_g.data)
        prg.mm9(queue, (ls,1), (ls,1), self.logits_g.data, self.res_g.data)
        prg.mm8(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), self.logits_g.data, self.res_g.data)
        prg.mm10(queue, (1,1), (1,1), self.logits_g.data)
        prg2.mm11(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), self.logits_g.data)
        prg.mm9(queue, (ls,1), (ls,1), self.logits_g.data, self.res_g.data)
        cl.enqueue_copy(queue, self.res, self.res_g.data)
        return self.res

    def kernel_3(self,x_g,weight_g,bias_g,attn_weight_g,attn_bias_g,new_cache_g\
        ,ln_f_weight_g,ln_f_bias_g,n_tokens,max_content,lm_head_weight_g,temperature,random_num):
        ls = 256
        size = self.dim #todo hardcoded
        b_cols2 = 50257
        b_rows2 = self.dim
        seg2 = math.ceil(50257 / ls)
        b_cols = self.dim*3 #todo
        b_rows = self.dim
        seg = int(size / ls) #todo
        x0_g = create_cl_buffer_empty(n_tokens*self.dim*4)
        logits_g = create_cl_buffer_empty(50257*4)
        c_g = create_cl_buffer_empty(n_tokens*b_cols*4)
        if hasattr(self, 'res') == False:
            self.res = np.zeros(1).astype(np.float32)
        res_g = create_cl_buffer_empty(1*4)
        prg_str = f"""
        __kernel void mm(__global const float *x_in,
            __global float *x, __global const float *weight, __global const float *bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{n_tokens}];
            int gidx0 = get_global_id(0);
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; 
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + lidx0*{seg} + i] = x_in[{self.dim}*r + lidx0*{seg} + i];
                temp2[r] += x[{self.dim}*r + lidx0*{seg} + i];
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
                x[{self.dim}*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[{self.dim}*r + lidx0*{seg} + i],2);
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
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
            }}
        }}
        __kernel void mm2(
            __global const float *x, __global const float *attn_weight, __global const float *attn_bias,__global float *res)
        {{
        if(get_global_id(0) < {b_cols*n_tokens}) {{ //TODO
            int gidx0 = get_global_id(0);
            int i = gidx0 / {n_tokens};
            int y = gidx0 % {n_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += x[y*{b_rows} + k] * attn_weight[i*{b_rows} + k]; 
            }}
            res[y*{b_cols} + i] = total + attn_bias[i];
        }}
        }}
        __kernel void mm3(
            __global const float *xqkv, __global float *new_cache)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim*1} + j];
            new_cache[{max_content*self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*2 + j]; 
        }}
         __kernel void mm4(
            __global const float *xqkv, __global float *new_cache)
        {{
            int gidx0 = get_global_id(0);
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}];
            new_cache[{max_content*self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}*2]; 
        }}
        __kernel void mm5(
            __global float *x, __global const float *ln_f_weight, __global const float *ln_f_bias)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float mean;
            int lidx0 = get_global_id(0);
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
                x[i + lidx0*{seg} + {(n_tokens - 1)*self.dim}] -= mean;
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(x[lidx0*{seg} + i + {(n_tokens - 1)*self.dim}],2);
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
                x[i + lidx0*{seg}] = (x[i + lidx0*{seg} + {(n_tokens - 1)*self.dim}] * ln_f_weight[i + lidx0*{seg}]) / mean + ln_f_bias[i + lidx0*{seg}];
            }}
        }}
        __kernel void matmul(
            __global const float *a, __global const float *b, __global float *res)
        {{
            int x = get_global_id(0);
            if(x < {b_cols2}) {{
                float total = 0;
                for(int k = 0; k < {b_rows2}; k++) {{
                    total += a[k] * b[x*{b_rows2} + k]; 
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
            int gidx0 = get_global_id(0);
            a[gidx0] = exp(a[gidx0] - res[0]);
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
            int lidx0 = get_global_id(0);
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
        prg.mm(queue, (ls*n_tokens,1), (ls,1),x_g.data, x0_g.data, weight_g.data, bias_g.data)
        prg.mm2(queue, (math.ceil((b_cols*n_tokens / ls)*ls),1), (ls,1), x0_g.data, attn_weight_g.data,attn_bias_g.data,c_g.data)
        ls = 256
        prg.mm4(queue, (math.ceil((n_tokens*self.n_heads*64) / ls) * ls,1), (ls,1), c_g.data, new_cache_g.data) 
        prg.mm5(queue, (ls,1), (ls,1), x_g.data, ln_f_weight_g.data, ln_f_bias_g.data)

        group_size = math.ceil(b_cols2 / ls) * ls
        prg.matmul(queue, (group_size,1), (ls,1), x_g.data, lm_head_weight_g.data,logits_g.data)
        prg.mm6(queue, (1,1), (1,1), logits_g.data, res_g.data) 
        prg.mm7(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g.data, res_g.data)
        prg.mm6(queue, (1,1), (1,1), logits_g.data, res_g.data)
        prg.mm9(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g.data, res_g.data)
        prg.mm10(queue, (ls,1), (ls,1), logits_g.data, res_g.data)
        prg.mm9(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g.data, res_g.data)
        prg.mm11(queue, (1,1), (1,1), logits_g.data)
        prg.mm12(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), logits_g.data)
        prg.mm10(queue, (ls,1), (ls,1), logits_g.data, res_g.data)
        cl.enqueue_copy(queue, self.res, res_g.data)
        return self.res

    def kernel_0(self,a_g,c_g,d_g,e_g,xqkv_g,g,keys_values_g,start_pos,weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g):
        ls = 256
        seg = int(self.dim / ls) #todo
        seg3 = math.ceil(self.n_heads*(start_pos+1)*(start_pos+1) / ls)
        if hasattr(self, 'temp_g') == False:
            self.temp_g = create_cl_buffer_empty(self.n_heads*self.max_context*4)
        if hasattr(self, 'xq_temp_g') == False:
            self.xq_temp_g = create_cl_buffer_empty(self.dim*4)
        prg_str = f"""
        __kernel void mm(
            __global float *a, __global float *mean)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            int lidx0 = get_global_id(0);
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
                mean[0] = total / {self.dim};  
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                a[i + lidx0*{seg}] -= mean[0];
                total += pow(a[lidx0*{seg} + i],2);
            }}
            temp[lidx0] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean[0] = pow(total / {self.dim} + 1e-5,0.5);
            }}
            }}        
        """
        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg = self.prg_cache[prg_str]

        prg_str = f"""
        __kernel void mm4(
            __global float *a, __global const float *c, __global const float *d, __global const float *e,
            __global const float *xqkv, __global float *keys_values,
            __global float *xq_temp, __global const float *mean)
        {{
            int lidx0 = get_global_id(0) % {ls};
            int i = get_global_id(0) / {ls};
            xq_temp[lidx0*{int(self.dim*3 / ls)} + i] = xqkv[lidx0*{int(self.dim*3 / ls)} + i];
            float total = 0;
            for(int k = 0; k < {self.dim}; k++) {{
                total += ((a[k] * c[k]) / mean[0] + d[k]) * e[(lidx0*{int(self.dim*3 / ls)} + i)*{self.dim} + k];
            }}
            if((lidx0*{int(self.dim*3 / ls)} + i) < {g}) {{
                xq_temp[lidx0*{int(self.dim*3 / ls)} + i] += total;
                }}
            if((lidx0*{int(self.dim*3 / ls)} + i) >= {g} && (lidx0*{int(self.dim*3 / ls)} + i) < {2*g}) {{
                keys_values[{start_pos}*{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {g}] = xqkv[{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {g}] + total;
            }}
            if((lidx0*{int(self.dim*3 / ls)} + i) >= {2*g}) {{
                keys_values[{self.dim*self.max_context} + {start_pos}*{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {2*g}] = xqkv[{self.dim*2} + lidx0*{int(self.dim*3 / ls)} + i - {2*g}] + total;
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
                    acc0 += xq_temp[i + 64*k] * keys_values[x*{self.n_heads*64} + i + 64*k];
                }}                  
                temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(64)
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
            __attribute__ ((aligned (16))) __local float bias3_temp[{self.dim*4}];
            __attribute__ ((aligned (16))) __local float bias4_temp[{self.dim*3}];
            __attribute__ ((aligned (16))) __local float h_temp[{self.dim}];
            __attribute__ ((aligned (16))) __local float h[{self.dim}];
            int lidx0 = get_global_id(0);
            if(lidx0 < {self.n_heads}){{
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
                    acc0 += temp3[i + {start_pos+1}*y] * keys_values[{self.dim*self.max_context} + i*{self.n_heads*64} + x + y*64];
                }}
                xq_temp[x + y*64] = acc0;
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {seg}; i++) {{
                float acc = 0;
                for(int x = 0; x < {self.dim}; x++) {{
                    acc += xq_temp[x] * weight[x*{self.dim} + lidx0*{seg} + i];
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
                mean = total / {self.dim};  
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
                mean = pow(total / {self.dim} + 1e-5,0.5);
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i = 0; i < {int(self.dim*4 / ls)}; i++) {{
                bias3_temp[i + lidx0*{int(self.dim*4 / ls)}] = bias3[i + lidx0*{int(self.dim*4 / ls)}];
                for(int j = 0; j < {self.dim}; j++) {{
                    bias3_temp[i + lidx0*{int(self.dim*4 / ls)}] += ((h[j] * weight2[j]) / mean + bias2[j]) * weight3[(i + lidx0*{int(self.dim*4 / ls)})*{self.dim} + j];
                }}
                bias3_temp[i + lidx0*{int(self.dim*4 / ls)}] = 0.5 * bias3_temp[i + lidx0*{int(self.dim*4 / ls)}]\
                * (1 + tanh(bias3_temp[i + lidx0*{int(self.dim*4 / ls)}] * 0.7978845608\
                * (1 + 0.044715 * pow(bias3_temp[i + lidx0*{int(self.dim*4 / ls)}],2))));
            }}
            barrier(CLK_LOCAL_MEM_FENCE);  
            for(int i = 0; i < {int(self.dim / ls)}; i++) {{
                bias4_temp[lidx0 + i*{ls}] = bias4[lidx0 + i*{ls}];
                for(int j = 0; j < {self.dim*4}; j++) {{
                    bias4_temp[lidx0 + i*{ls}] += bias3_temp[j] * weight4[lidx0 + i*{ls} + j*{self.dim}];
                }}
                a[lidx0 + i*{ls}] = bias4_temp[lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
            }}
        }}
        """

        if prg_str not in self.prg_cache:
            self.prg_cache[prg_str] = cl.Program(ctx,prg_str).build()
        prg2 = self.prg_cache[prg_str]

        if hasattr(self, 'mean') == False:
            self.mean = create_cl_buffer_empty(1*4)

        prg.mm(queue, (ls,1), (ls,1),a_g.data,self.mean.data)
        prg2.mm4(queue, (self.dim*3,1), (ls,1),a_g.data,c_g.data,d_g.data,e_g.data,xqkv_g.data\
        ,keys_values_g.data,self.xq_temp_g.data,self.mean.data)
        prg2.mm2(queue, (ls*seg3,1), (ls,1),keys_values_g.data,self.temp_g.data, self.xq_temp_g.data)
        prg2.mm3(queue, (ls,1), (ls,1),a_g.data\
        ,keys_values_g.data,weight_g.data,bias_g.data,\
        weight2_g.data,bias2_g.data,weight3_g.data,bias3_g.data,weight4_g.data,bias4_g.data,self.temp_g.data, self.xq_temp_g.data)
        return a_g
        
    def kernel_2(self,x_g,ln_1_weight_g,ln_1_bias_g,attn_weight_g,attn_bias_g,cache_kv_g,attn_c_proj_weight_g,attn_c_proj_bias_g,ln_2_weight_g,ln_2_bias_g,c_fc_weight_g,c_fc_bias_g\
        ,c_proj_weight_g,c_proj_bias_g,num_tokens,max_content):
        if hasattr(self, 'h_g') == False:
            self.h_g = create_cl_buffer_empty(max_content*self.dim*4)
        if hasattr(self, 'h2_g') == False:
            self.h2_g = create_cl_buffer_empty(max_content*self.dim*4)
        if hasattr(self, 'xq_g') == False:
            self.xq_g = create_cl_buffer_empty(self.n_heads*64*max_content*4)
        if hasattr(self, 'xv_g') == False:
            self.xv_g = create_cl_buffer_empty(self.n_heads*64*max_content*4)
        if hasattr(self, 'c_g') == False:
            self.c_g = create_cl_buffer_empty(self.n_heads*64*max_content*4)
        if hasattr(self, 'xqt_g') == False:
            self.xqt_g = create_cl_buffer_empty(self.n_heads*64*max_content*4)
        if hasattr(self, 'res_g') == False:
            self.res_g = create_cl_buffer_empty(max_content*self.n_heads*4)
        if hasattr(self, 'xqkv_g') == False:
            self.xqkv_g = create_cl_buffer_empty(max_content*self.dim*3*4)
        if hasattr(self, 'd_g') == False:
            self.d_g = create_cl_buffer_empty(max_content*self.dim*4*4)
        a_rows = num_tokens
        a_cols = 64
        b_rows = self.dim
        ls = 256
        size = self.dim
        seg = int(size / ls) #todo
        b_cols = self.dim*3 # for first part
        b_cols_2 = self.dim*4
        prg = cl.Program(ctx, f"""
        __kernel void mm(
            __global float *x, __global const float *weight, __global const float *bias,
            __global float *copy)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{num_tokens}];
            int gidx0 = get_global_id(0);
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; //todo clean
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                copy[{self.dim}*r + lidx0*{seg} + i] = x[{self.dim}*r + lidx0*{seg} + i];
                temp2[r] += x[{self.dim}*r + lidx0*{seg} + i];
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
                x[{self.dim}*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[{self.dim}*r + lidx0*{seg} + i],2);
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
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
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
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*1 + j];
            new_cache[{max_content}*{self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*2 + j]; 
        }}                 
        __kernel void tr(
            __global const float *xqkv, __global float *xq, __global float *xv)
        {{
            int gidx0 = get_global_id(0);
            int i = (gidx0 / {64}) / {num_tokens};
            int j = (gidx0 / {64}) % {num_tokens};
            int k = gidx0 % 64;
            xq[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k];
            xv[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k + {64*self.n_heads*2}];
        }}
        __kernel void ms0(
            __global float *xq, __global const float *xqkv)
        {{
            int gidx0 = get_global_id(0);
            if(gidx0 < {self.n_heads*a_rows*a_rows}) {{
                int x = (gidx0 / {a_rows}) % {a_rows};
                int z = gidx0 / ({a_rows}*{a_rows}); 
                int y = gidx0 % {a_rows};
                float total = 0;
                for(int k = 0; k < {a_cols}; k++) {{
                    total += xq[y*{a_cols} + k + z*{a_rows}*{a_cols}] * xqkv[x*{64*self.n_heads*3} + k + z*64 + {self.dim}]; 
                }}
                xq[y*{a_rows} + x + z*{a_rows}*{a_rows}] = total / 8; //sqrt 64 input shape xq
            }}
        }}
        __kernel void ms(
            __global float *xq)
        {{
        int gidx0 = get_global_id(0);
        if(gidx0 < {num_tokens*num_tokens*self.n_heads}) {{
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
        if(gidx0 < {num_tokens*self.n_heads}) {{
        int x = gidx0 / {num_tokens};
        int y = gidx0 % {num_tokens};
            float m = 0;
            for(int z = 0; z < {num_tokens}; z++) {{
                m += xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z];
            }}
            mx[x*{num_tokens} + y] = m;  
            }}
        }}
        __kernel void ms4(
            __global float *xq, global const float *mx)
        {{
        int gidx0 = get_global_id(0);
        if(gidx0 < {num_tokens*num_tokens*self.n_heads}) {{
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
            xqt[i*64 + j*{self.n_heads*64} + k] = xq[i*{num_tokens}*64 + j*64 + k];
        }}
        __kernel void ms7(
            __global const float *xq, __global const float *attn_weight,__global const float *attn_bias, __global float *res)
        {{
            int gidx0 = get_global_id(0);
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += xq[y*{b_rows} + k] * attn_weight[x*{b_rows} + k]; 
            }}
            res[y*{b_rows} + x] += total + attn_bias[x];
        }}
        __kernel void ms8(
            __global float *x, __global const float *ln_2_weight, __global const float *ln_2_bias
            ,__global float *copy)
        {{
            __attribute__ ((aligned (16))) __local float temp[{ls}];
            __attribute__ ((aligned (16))) __local float temp2[{num_tokens}];
            int gidx0 = get_global_id(0);
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; //todo clean
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                copy[{self.dim}*r + lidx0*{seg} + i] = x[{self.dim}*r + lidx0*{seg} + i];
                temp2[r] += x[{self.dim}*r + lidx0*{seg} + i];
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
                x[{self.dim}*r + i + lidx0*{seg}] -= temp2[r];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[{self.dim}*r + lidx0*{seg} + i],2);
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
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * ln_2_weight[i + lidx0*{seg}]) / temp2[r] + ln_2_bias[i + lidx0*{seg}];
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
        """).build()
        prg.mm(queue, (ls*num_tokens,1), (ls,1), x_g.data, ln_1_weight_g.data, ln_1_bias_g.data,self.h_g.data) 
        g = math.ceil((b_cols*num_tokens / ls)*ls)
        prg.mm2(queue, (g,1), (ls,1), x_g.data, attn_weight_g.data,attn_bias_g.data,self.xqkv_g.data)
        g = math.ceil((num_tokens*self.n_heads*64) / ls) * ls
        prg.mm3(queue, (g,1), (ls,1), self.xqkv_g.data, cache_kv_g.data)

        ls = 256
        prg.tr(queue, (g,1), (ls,1), self.xqkv_g.data, self.xq_g.data, self.xv_g.data)

        g = math.ceil(self.n_heads*num_tokens*num_tokens / ls) * ls
        prg.ms0(queue, (g,1), (ls,1), self.xq_g.data, self.xqkv_g.data)
        prg.ms(queue, (g,1), (ls,1), self.xq_g.data)
        if self.n_heads*num_tokens > ls:
            g2 =  math.ceil(self.n_heads*num_tokens / ls) * ls
        else:
            g2 = self.n_heads*num_tokens
        prg.ms3(queue, (g2,1), (min(self.n_heads*num_tokens,ls),1), self.xq_g.data,self.res_g.data)
        prg.ms4(queue, (g,1), (ls,1), self.xq_g.data,self.res_g.data)

        
        g3 = (math.ceil(self.n_heads*a_cols*num_tokens / ls) * ls)
        prg.ms5(queue, (g3,1), (ls,1), self.xq_g.data,self.xv_g.data,self.c_g.data)
        g4 = num_tokens*self.n_heads*64
        g4 = math.ceil(g4 / ls)*ls
        
        prg.ms6(queue, (g4,1), (ls,1), self.c_g.data,self.xqt_g.data)

        g = math.ceil((b_rows*num_tokens / ls)*ls)
        prg.ms7(queue, (g,1), (ls,1), self.xqt_g.data,attn_c_proj_weight_g.data,attn_c_proj_bias_g.data,self.h_g.data)
        prg.ms8(queue, (ls*num_tokens,1), (ls,1), self.h_g.data, ln_2_weight_g.data, ln_2_bias_g.data,self.h2_g.data)

        g = math.ceil((b_cols_2*num_tokens / ls)*ls)
        prg.ms9(queue, (g,1), (ls,1), self.h_g.data, c_fc_weight_g.data,c_fc_bias_g.data,self.d_g.data)


        g = math.ceil((b_rows*num_tokens / ls)*ls)
        prg.ms10(queue, (g,1), (ls,1), self.d_g.data, c_proj_weight_g.data,c_proj_bias_g.data,self.h2_g.data)
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
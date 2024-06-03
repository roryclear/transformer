	

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

def add(a,b,b_s=0,a_s=0):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    res_np = np.zeros(768).astype(np.float32).flatten()
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void add(
        __global const float *a, __global const float *b, __global float *res)
    {{
    int gidx0 = get_global_id(0);
        res[gidx0] = a[{int(a_s)*768} + gidx0] + b[gidx0 + {b_s}*768];   
    }}
    """).build()
    knl = prg.add
    knl(queue, (768,1), (256,1), a_g, b_g,res_g) #todo check shape
    return res_g

len = 768
loop_size = int(len / 256)
len_short = 768

def kernel_6(a_g,random_num):
    res = np.zeros(1).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    ls = 256
    seg = math.ceil(50257 / ls)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global const float *a, __global float *res)
    {{
        res[0] = a[0]; //todo why is this needed?, used to be a MAX
    }}

    __kernel void mm2(
    __global float *a, __global const float *res)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);
        int gidx0 = get_global_id(0);
        a[gidx0] = exp(a[gidx0] - res[0]);
    }}

    __kernel void mm3(
        __global const float *a, __global float *res)
    {{
        res[0] = a[0];
    }}

    __kernel void mm4(
    __global float *a, __global const float *res)
    {{
        int gidx0 = get_global_id(0);
        a[gidx0] = a[gidx0] / res[0];
    }}

    __kernel void mm5(
    __global const float *a, __global float *res)
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
            res[0] = t;
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
    """).build()
    knl = prg.mm
    knl(queue, (1,1), (1,1), a_g, res_g) 
    knl2 = prg.mm2
    knl2(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), a_g, res_g)
    knl3 = prg.mm3
    knl3(queue, (1,1), (1,1), a_g, res_g)
    knl4 = prg.mm4
    knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), a_g, res_g)
    knl5 = prg.mm5
    knl5(queue, (ls,1), (ls,1), a_g, res_g)
    knl4(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), a_g, res_g)
    knl6 = prg.mm6
    knl6(queue, (1,1), (1,1), a_g)
    knl7 = prg.mm7
    knl7(queue, (math.ceil(50257 / ls)*ls,1), (ls,1), a_g)
    knl5(queue, (ls,1), (ls,1), a_g, res_g)
    cl.enqueue_copy(queue, res, res_g)
    #print(res)
    ##
    #cl.enqueue_copy(queue, a, a_g)
    return res

def tok_emb(tokens,weight,weight2):
    tokens_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tokens)
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    weight_2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    no_tokens = np.shape(tokens)[0]
    tok_emb = np.zeros((no_tokens,dim)).astype(np.float32)
    ls = 256
    size = no_tokens*dim

    tok_emb_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tok_emb)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global int *tokens, __global const float *weight, __global const float *weight2,  __global float *tok_emb)
    {{
        int gidx0 = get_global_id(0);
        int i = gidx0 / {dim};
        int j = gidx0 % {dim};
        tok_emb[i*{dim} + j] = weight[tokens[i]*{dim} + j] + weight2[i*{dim} + j];
    }}
    """).build()
    knl = prg.mm
    knl(queue, (math.ceil(size / ls)*ls,1), (ls,1), tokens_g, weight_g, weight_2_g,tok_emb_g)
    cl.enqueue_copy(queue, tok_emb, tok_emb_g)
    return tok_emb

def minus_mean_multi(a):
    size = np.shape(a)[0]
    seg = int(size / 32) #todo
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg}];
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
            for(int i = 0; i < 32; i++) {{ //rory todo 32
                total += temp[i];
            }}
            mean = total / {size};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] -= mean;
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (32,1), (32,1), a_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    return a

def kernel_3(h_g,weight_g,bias_g):
    ls = 256
    seg = int(dim / ls) #todo
    prg = cl.Program(ctx, f"""
    __kernel void mm4(
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
    """).build()
    knl = prg.mm4
    knl(queue, (ls,1), (ls,1), h_g, weight_g, bias_g) #rory to test large stuff
    return h_g

def kernel_0(a,c,d):
    size = np.shape(a)[0]
    ls = 256
    seg = int(size / ls) #todo
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg}];
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
            mean = total / {size};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] -= mean;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            total += pow(a[lidx0*{seg} + i],2);
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
            a[i + lidx0*{seg}] = (a[i + lidx0*{seg}] * c[i + lidx0*{seg}]) / mean + d[i + lidx0*{seg}];
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1), a_g, c_g, d_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    return a

def kernel_0_12(a,c,d):
    size = 768 #todo hardcoded
    ls = 256
    seg = int(size / ls) #todo
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d)
    {{
        __attribute__ ((aligned (16))) __local float temp[{ls}];
        __attribute__ ((aligned (16))) __local float temp2[12];
        int lidx0 = get_local_id(0);
        temp2[0] = 0;
        for(int i = 0; i < {seg}; i++) {{
            temp2[0] += a[lidx0*{seg} + i];
        }}
        temp[lidx0] = temp2[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            temp2[0] = 0;
            for(int i = 0; i < {ls}; i++) {{
                temp2[0] += temp[i];
            }}
            temp2[0] = temp2[0] / {size};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] -= temp2[0];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        temp2[0] = 0;
        for(int i = 0; i < {seg}; i++) {{
            temp2[0] += pow(a[lidx0*{seg} + i],2);
        }}
        temp[lidx0] = temp2[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            temp2[0] = 0;
            for(int i = 0; i < {ls}; i++) {{
                temp2[0] += temp[i];
            }}
            temp2[0] = pow(temp2[0] / {size} + 1e-5,0.5);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] = (a[i + lidx0*{seg}] * c[i + lidx0*{seg}]) / temp2[0] + d[i + lidx0*{seg}];
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1), a_g, c_g, d_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    a = a.flatten()[:768]
    return a

def kernel_0_b(x,weight,bias,n_tokens):
    size = 768 #todo hardcoded
    ls = 256
    seg = int(size / ls) #todo
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *x, __global const float *weight, __global const float *bias)
    {{
        __attribute__ ((aligned (16))) __local float temp[{ls}];
        __attribute__ ((aligned (16))) __local float temp2[{n_tokens}];
        int lidx0 = get_local_id(0);
        int gidx0 = get_group_id(0);
        int r = gidx0; //todo clean
        temp2[r] = 0;
        for(int i = 0; i < {seg}; i++) {{
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
    """).build()
    knl = prg.mm
    knl(queue, (ls*n_tokens,1), (ls,1), x_g, weight_g, bias_g) #rory to test large stuff
    return x_g

def kernel_2(a_g,c_g,d_g,e_g,xqkv_g,g,keys_values_g,start_pos,weight_g,bias_g,\
    weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g): #g = size
    ls = 256
    xq_temp = np.zeros(768).astype(np.float32)
    zeros2 = np.zeros(12*(start_pos+1)).astype(np.float32)
    seg = int(dim / ls) #todo
    seg3 = math.ceil(12*(start_pos+1)*(start_pos+1) / ls)
    temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros2)
    xq_temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq_temp)
    prg = cl.Program(ctx, f"""
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
    
    """).build()

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

def kernel_4(h,c,d,f,g,start_pos,bias,\
    weight2,bias2,bias3,\
    e,keys,values,weight,weight3,weight4,bias4,keys_1,values_1): #g = size
    ls = 256
    zeros = np.zeros(np.shape(bias4)[0]).astype(np.float32)
    zeros2 = np.zeros(12*(start_pos+1)).astype(np.float32)
    seg = int(dim / ls) #todo
    seg3 = math.ceil(12*(start_pos+1)*(start_pos+1) / ls)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    e_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e)
    keys_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    weight2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias2)
    weight3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight3)
    bias3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias3)
    weight4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight4)
    bias4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias4)

    keys_g_1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys_1)
    values_g_1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values_1)

    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    h_temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros2)
    xqkv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d, __global const float *e,
        __global float *xqkv, __global float *keys,
        __global float *values,
        __global const float *weight,__global const float *bias,
        __global const float *weight2, __global const float *bias2,
        __global const float *weight3, __global float *bias3,
        __global const float *weight4,
        __global float *bias4, __global float *h_temp, __global float *h,
        __global float *temp3,
        __global float *keys_1,
        __global float *values_1)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg}];
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
            float total = 0;
            for(int k = 0; k < {dim}; k++) {{
                total += ((a[k] * c[k]) / mean + d[k]) * e[(lidx0*{int(dim*3 / ls)} + i)*{dim} + k];
            }}
            if((lidx0*{int(dim*3 / ls)} + i) < {g}) {{
                xqkv[lidx0*{int(dim*3 / ls)} + i] += total;
                }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {g} && (lidx0*{int(dim*3 / ls)} + i) < {2*g}) {{
                keys[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {g}] = xqkv[768 + lidx0*{int(dim*3 / ls)} + i - {g}] + total;
            }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {2*g}) {{
                values[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {2*g}] = xqkv[768*2 + lidx0*{int(dim*3 / ls)} + i - {2*g}] + total;
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int z = 0; z < {seg3}; z++) {{
            int x = (z + lidx0*{seg3}) % {start_pos+1};
            int k = (z + lidx0*{seg3}) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xqkv[i + 64*k] * keys[x*12*64 + i + 64*k];
            }}                  
            temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
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
                acc0 += temp3[i + {start_pos+1}*y] * values[i*12*64 + x + y*64];
            }}
            xqkv[x + y*64] = acc0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            float acc = 0;
            for(int x = 0; x < {dim}; x++) {{
                acc += xqkv[x] * weight[x*{dim} + lidx0*{seg} + i];
            }}
            h[lidx0*{seg} + i] = a[lidx0*{seg} + i] + acc + bias[lidx0*{seg} + i];
            h_temp[lidx0*{seg} + i] = h[lidx0*{seg} + i];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
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
            for(int j = 0; j < {dim}; j++) {{
                bias3[i + lidx0*{int(dim*4 / ls)}] += ((h[j] * weight2[j]) / mean + bias2[j]) * weight3[(i + lidx0*{int(dim*4 / ls)})*{dim} + j];
            }}
            bias3[i + lidx0*{int(dim*4 / ls)}] = 0.5 * bias3[i + lidx0*{int(dim*4 / ls)}]\
            * (1 + tanh(bias3[i + lidx0*{int(dim*4 / ls)}] * 0.7978845608\
            * (1 + 0.044715 * pow(bias3[i + lidx0*{int(dim*4 / ls)}],2))));
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  
        for(int i = 0; i < {int(np.shape(bias4)[0] / 2 / ls)}; i++) {{ //todo because there's two now!
            for(int j = 0; j < {dim*4}; j++) {{
                bias4[lidx0 + i*{ls}] += bias3[j] * weight4[lidx0 + i*{ls} + j*{dim}];
            }}
            a[lidx0 + i*{ls}] = bias4[lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
        }}
        
        //PT2

        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
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
            float total = 0;
            for(int k = 0; k < {dim}; k++) {{
                total += ((a[k] * c[768*1 + k]) / mean + d[768*1 + k]) * e[768*2304*1 + (lidx0*{int(dim*3 / ls)} + i)*{dim} + k];
            }}
            if((lidx0*{int(dim*3 / ls)} + i) < {g}) {{
                xqkv[2304*1 + lidx0*{int(dim*3 / ls)} + i] += total;
                }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {g} && (lidx0*{int(dim*3 / ls)} + i) < {2*g}) {{
                keys_1[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {g}] = xqkv[2304*1+768 + lidx0*{int(dim*3 / ls)} + i - {g}] + total;
            }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {2*g}) {{
                values_1[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {2*g}] = xqkv[2304*1+768*2 + lidx0*{int(dim*3 / ls)} + i - {2*g}] + total;
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int z = 0; z < {seg3}; z++) {{
            int x = (z + lidx0*{seg3}) % {start_pos+1};
            int k = (z + lidx0*{seg3}) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xqkv[2304*1 + i + 64*k] * keys_1[x*12*64 + i + 64*k];
            }}                  
            temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
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
                acc0 += temp3[i + {start_pos+1}*y] * values_1[i*12*64 + x + y*64];
            }}
            xqkv[2304*1 + x + y*64] = acc0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            float acc = 0;
            for(int x = 0; x < {dim}; x++) {{
                acc += xqkv[2304*1 + x] * weight[768*768*1 + x*{dim} + lidx0*{seg} + i];
            }}
            h[lidx0*{seg} + i] = a[lidx0*{seg} + i] + acc + bias[768*1 + lidx0*{seg} + i];
            h_temp[lidx0*{seg} + i] = h[lidx0*{seg} + i];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
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
            for(int j = 0; j < {dim}; j++) {{
                bias3[3072*1 + i + lidx0*{int(dim*4 / ls)}] += ((h[j] * weight2[768*1 + j]) / mean + bias2[768*1 + j]) * weight3[768*3072*1 + (i + lidx0*{int(dim*4 / ls)})*{dim} + j];
            }}
            bias3[3072*1 + i + lidx0*{int(dim*4 / ls)}] = 0.5 * bias3[3072*1 + i + lidx0*{int(dim*4 / ls)}]\
            * (1 + tanh(bias3[3072*1 + i + lidx0*{int(dim*4 / ls)}] * 0.7978845608\
            * (1 + 0.044715 * pow(bias3[3072*1 + i + lidx0*{int(dim*4 / ls)}],2))));
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  
        for(int i = 0; i < {int(np.shape(bias4)[0] / 2 / ls)}; i++) {{ //todo because there's 2 now
            for(int j = 0; j < {dim*4}; j++) {{
                bias4[768*1 + lidx0 + i*{ls}] += bias3[3072*1 + j] * weight4[3072*768*1 + lidx0 + i*{ls} + j*{dim}];
            }}
            a[lidx0 + i*{ls}] = bias4[768*1 + lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1),a_g,c_g,d_g,e_g,xqkv_g\
    ,keys_g,values_g,weight_g,bias_g,\
    weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,h_g,h_temp_g,temp_g
    ,keys_g_1,values_g_1)
    cl.enqueue_copy(queue, keys, keys_g)
    cl.enqueue_copy(queue, values, values_g)
    cl.enqueue_copy(queue, keys_1, keys_g_1)
    cl.enqueue_copy(queue, values_1, values_g_1)
    cl.enqueue_copy(queue, h, a_g)
    return h

def kernel_1b(h_in,c,d,f):
    size = np.shape(h_in)[0]
    ls = 256
    seg5 = int(size / ls)
    h_in_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_in)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    prg = cl.Program(ctx, f"""
    __kernel void knl(
        __global float *h_in, __global const float *c, __global const float *d,
        __global float *f)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg5}];
        __attribute__ ((aligned (16))) __local float mean;
        int lidx0 = get_local_id(0);
        float total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += h_in[lidx0*{seg5} + i];
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] -= mean;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += pow(h_in[lidx0*{seg5} + i],2);
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] = (h_in[i + lidx0*{seg5}] * c[i + lidx0*{seg5}]) / mean + d[i + lidx0*{seg5}];
        }}
    }}
    """).build()
    knl = prg.knl
    knl(queue, (ls,1), (ls,1), h_in_g, c_g, d_g,f_g)
    cl.enqueue_copy(queue, h_in, h_in_g)
    return h_in


def sq_mean_sqrt(a):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void sq_mean_sqrt(
        __global float *data0)
    {{
    float avg = 0;
    __attribute__ ((aligned (16))) __local float temp[256];
    int lidx0 = get_local_id(0); /* 256 */
    int gidx0 = get_global_id(0); /* 256 */
    float acc0 = 0.0f;
    for (int ridx0 = 0; ridx0 < {loop_size}; ridx0++) {{
        float val0 = data0[(lidx0*{loop_size})+ridx0];
        acc0 = (pow(val0,2)+acc0);
    }}
    temp[lidx0] = acc0;
    barrier(CLK_LOCAL_MEM_FENCE); //rory need this barrier when input is large like 7680, 768 seems to work
    //I think this woudlnt work if loop was smaller than len / 256?
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 256; ridx1++) {{
        float val1 = temp[ridx1];
        acc1 += val1;
    }}
    avg = acc1 / {len};
    data0[0] = pow(avg + 1e-5,0.5);
    }}
    """).build()
    knl = prg.sq_mean_sqrt
    knl(queue, (256,1), (256,1), a_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    return a[0]

def divide(a,b,c,d):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    prg = cl.Program(ctx, f"""
    __kernel void divide(
        __global float *a, __global const float *b, __global const float *c, __global const float *d)
    {{
    int gidx0 = get_global_id(0);
    a[gidx0] = (a[gidx0] * c[gidx0]) / b[0] + d[gidx0];
    }}
    """).build()
    knl = prg.divide

    knl(queue, (768,1), (256,1), a_g, b_g, c_g, d_g) #has to be multiple of 256
    cl.enqueue_copy(queue, a, a_g)
    return a

def matmul2(a,b,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten() #todo, shouldnt be needed
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([12,1,s])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
    int x = get_global_id(0) % {s};
    int k = get_global_id(0) / {s};
    float acc0 = 0.0f;
    float4 vals_a[16];
    for(int i = 0; i < 16; i++) {{
        vals_a[i] = (float4)(*((__global float4*)(a+i*4+64*k)));
    }}
                    
    float vals_b[64];
    for(int i = 0; i < 64; i++) {{
        vals_b[i] = b[x+i*{s} + {s}*64*k]; // rory i*x
    }}
    
    for(int i = 0; i < 16; i++) {{
        acc0 = mad((vals_a[i]).x,vals_b[i*4],acc0);
        acc0 = mad((vals_a[i]).y,vals_b[i*4 + 1],acc0);
        acc0 = mad((vals_a[i]).z,vals_b[i*4 + 2],acc0);
        acc0 = mad((vals_a[i]).w,vals_b[i*4 + 3],acc0);
    }}
    res[x + k*{s}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
    }}
    """).build()
    knl = prg.matmul
    local0 = min(256,s*64*12)
    group0 = math.ceil(12*s / local0) * local0
    knl(queue, (group0,1), (local0,1), a_g, b_g,c_g)
    cl.enqueue_copy(queue, c, c_g)
    return c

def minus_max(a,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c = np.zeros((12,1,s))
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void k(
        __global const float *data1, __global float *data0)
    {{
        int lid = get_local_id(0);
        float m = -INFINITY;
        for(int i = 0; i < {s}; i++) {{
            float val = data1[i + lid*{s}];
            m = max(m,val);
        }}
        for(int i = 0; i < {s}; i++) {{
            data0[i + lid*{s}] = exp(data1[i + lid*{s}] - m);
        }}
        barrier(CLK_LOCAL_MEM_FENCE); //not needed in practice?
        float t = 0;
        for(int i = 0; i < {s}; i++) {{
            float val = data0[i + lid*{s}];
            t = t+val;
        }}
        for(int i = 0; i < {s}; i++) {{
            data0[i + lid*{s}] = data0[i + lid*{s}] / t;
        }}
    }}
    """).build()
    knl = prg.k
    knl(queue, (12,1), (12,1), a_g,c_g) #todo hardcoded
    cl.enqueue_copy(queue, c, c_g)
    return c

def minus_max_b(a):
    s = np.shape(a)[2] #todo remove dim
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    ls = 256

    prg = cl.Program(ctx, f"""
    __kernel void k(
        __global float *data0)
    {{
        int lidx0 = get_local_id(0);
        if(lidx0 < 12){{
        float m = -INFINITY;
        for(int i = 0; i < {s}; i++) {{
            float val = data0[i + lidx0*{s}];
            m = max(m,val);
        }}
        for(int i = 0; i < {s}; i++) {{
            data0[i + lidx0*{s}] = exp(data0[i + lidx0*{s}] - m);
        }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE); //not needed in practice?
        if(lidx0 < 12) {{
        float t = 0;
        for(int i = 0; i < {s}; i++) {{
            float val = data0[i + lidx0*{s}];
            t = t+val;
        }}
        for(int i = 0; i < {s}; i++) {{
            data0[i + lidx0*{s}] = data0[i + lidx0*{s}] / t;
        }}
        }}
    }}
    """).build()
    knl = prg.k
    knl(queue, (ls,1), (ls,1), a_g) #todo hardcoded
    cl.enqueue_copy(queue, a, a_g)
    return a

def matmul3(a,b,s):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten() #todo, shouldnt be needed
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([12,1,64])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int lidx0 = get_local_id(0);
        int gidx0 = get_group_id(0);
        float acc0 = 0;
        float4 vals_a[{s}];
        for(int i = 0; i < {s}; i++) {{
            vals_a[i] = a[i + {s}*gidx0];
        }}
        float vals_b[{s}];
        for(int i = 0; i < {s}; i++) {{
            vals_b[i] = b[i*64 + lidx0 + gidx0*{s}*64];
        }}
        for(int i = 0; i < {s}; i++) {{
            acc0 = mad(vals_a[i].x,vals_b[i],acc0);
        }}
        res[lidx0 + gidx0*64] = acc0;
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (64*12,1), (64,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul3_b(xq,values,s):
    xq = xq.flatten()
    xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
    values = values.flatten() #todo, shouldnt be needed
    values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    res = np.zeros([12,1,64]).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    ls = 256
    seg = int((12*64) / ls)
    seg2 = math.ceil((np.shape(values)[0] / 64) / 256)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *xq, __global const float *values, __global float *res)
    {{
        int lidx0 = get_local_id(0);
        for(int g = 0; g < {seg}; g++) {{
            int y = (g + lidx0*{seg}) / 64;
            int x = (g + lidx0*{seg}) % 64;
            float acc0 = 0;
            for(int i = 0; i < {s}; i++) {{
                acc0 += xq[i + {s}*y] * values[i*64 + x + y*{s}*64];
            }}
            res[x + y*64] = acc0;
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (ls,1), (ls,1), xq_g, values_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

def transpose_f(a):
    s = np.shape(a)[0]    
    a = a.flatten()
    at = np.zeros_like(a)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    at_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=at)
    seg2 = math.ceil((np.shape(a)[0] / 64) / 256)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global float *at)
    {{
        int lidx0 = get_local_id(0);
        for(int i = 0; i < {seg2}; i++) {{
            int y = (lidx0*{seg2} + i) / 12;
            int x = (lidx0*{seg2} + i) % 12;
            for(int k = 0; k < 64; k++) {{
                if((y*12*64 + x*64 + k) < {np.shape(a)[0]}) {{
                    //at[x*64*{s} + y*64 + k] = a[y*12*64 + x*64 + k];
                    at[x*64*{s} + y + k*{s}] = a[y*12*64 + x*64 + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (256,1), (256,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at.reshape(12,64,s)
    #return at.reshape(12,s,64)

def matvec(a,b,c):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros([768])
    d = np.float32(d)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b, __global const float *c, __global float *res)
    {{
        int lidx0 = get_global_id(0);
        float acc = 0;
        for(int x = 0; x < 768; x++) {{
            acc += a[x] * b[x*768 + lidx0];
        }}
        res[lidx0] = acc + c[lidx0];
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (768,1), (256,1), a_g, b_g,c_g,d_g)
    cl.enqueue_copy(queue, d, d_g)
    return d

def matvec_b(a,b,c,h):
    ls = 256
    a = a.flatten()
    b = b.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    s = np.shape(a)[0]
    seg = int(np.shape(a)[0] / ls)
    print("s and seg ->",s,seg)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b, __global const float *c,
        __global float *h)
    {{
        int lidx0 = get_local_id(0);
        for(int i = 0; i < {seg}; i++) {{
            float acc = 0;
            for(int x = 0; x < {s}; x++) {{
                acc += a[x];// * b[x*{s} + lidx0*{seg} + i];
            }}
            h[lidx0*{seg} + i] += acc + c[lidx0*{seg} + i];
        }}
    h[0] = b[0];
    h[1] = b[1];
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (ls,1), (ls,1), a_g, b_g,c_g,h_g)
    cl.enqueue_copy(queue, h, h_g)
    return h

def matvec2(h_g,weight2_g,temperatue): #pass bias in instead of adding to zero, todo for other kernels
    rows = 768
    cols = 50257
    res = np.zeros(cols).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *h, __global const float *weight2 , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int j = 0; j < {rows}; j++) {{
            res[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
        }}
        res[gidx0] /= {temperatue};
    }}
    """).build()
    knl = prg.matvec
    gidx = math.ceil(cols / 16) * 16
    knl(queue, (gidx,1), (16,1), h_g, weight2_g,res_g)
    return res_g

def matvec2_notemp(h_g,weight2_g): #pass bias in instead of adding to zero, todo for other kernels
    rows = 768
    cols = 50257
    res = np.zeros(cols).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *h, __global const float *weight2 , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int j = 0; j < {rows}; j++) {{
            res[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
        }}
    }}
    """).build()
    knl = prg.matvec
    gidx = math.ceil(cols / 16) * 16
    knl(queue, (gidx,1), (16,1), h_g, weight2_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

def matvec2_b(h,weight2): #pass bias in instead of adding to zero, todo for other kernels
    rows = 768
    cols = 50257
    res = np.zeros(cols).astype(np.float32)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    d = 1
    gidx = math.ceil((cols/d) / 256) * 256
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *h, __global const float *weight2 , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int i = 0; i < {d}; i++) {{
            for(int j = 0; j < {rows}; j++) {{
                res[gidx0 + {math.ceil((cols/d) / 256) * 256}*i] += 
                h[j] * weight2[gidx0 + {math.ceil((cols/d) / 256) * 256}*i + j*{cols}];
            }}
        }}
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (gidx,1), (256,1), h_g, bias2_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

def matmul_t(a,b):
    a_rows = np.shape(a)[0]
    b_cols = np.shape(b)[1]
    b_rows = np.shape(b)[0]
    c = np.zeros([a_rows,b_cols])
    ls = 256
    ####TRANSPOSED, this replicates it for a test. todo: fix 
    '''
    b2 = np.copy(b)
    b = np.empty((np.shape(b2)[1],np.shape(b2)[0]),dtype=np.float32)
    print("SHAPE =",np.shape(b)) 
    for j in range(np.shape(b)[0]):
        for i in range(np.shape(b)[1]):
            b[j][i] = np.copy(b2[i][j])
    '''
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        int x = gidx0 / {a_rows};
        int y = gidx0 % {a_rows};
        float total = 0;
        for(int k = 0; k < {b_rows}; k++) {{
            total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
        }}
        res[y*{b_cols} + x] = total;
    }}
    """).build()
    g = math.ceil((b_cols*a_rows / ls)*ls)
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_b(a_g,b,n_tokens,bias_g):
    a_rows = n_tokens
    b_cols = np.shape(b)[1]
    b_rows = np.shape(b)[0]
    c = np.zeros([a_rows,b_cols])
    ####TRANSPOSED
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    #bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global const float *bias, __global float *res)
    {{
        int x = get_global_id(0);
        if(x < {b_cols}) {{
            for(int y = 0; y < {a_rows}; y++) {{
                float total = 0;
                for(int k = 0; k < {b_rows}; k++) {{
                    total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
                }}
                res[y*{b_cols} + x] = bias[x] + total;
            }}  
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(b_cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,bias_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_c(a,b):
    b_cols = np.shape(b)[1]
    b_rows = np.shape(b)[0]
    c = np.zeros(b_cols)
    ####TRANSPOSED, this replicates it for a test. todo: fix 
    '''
    b2 = np.copy(b)
    b = np.empty((np.shape(b2)[1],np.shape(b2)[0]),dtype=np.float32)
    print("SHAPE =",np.shape(b)) 
    for j in range(np.shape(b)[0]):
        for i in range(np.shape(b)[1]):
            b[j][i] = np.copy(b2[i][j])
    '''
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int x = get_global_id(0);
        if(x < {b_cols}) {{
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += a[k] * b[x*{b_rows} + k]; 
            }}
            res[x] = total; 
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(b_cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_3d(a,b):
    a_rows = np.shape(a)[1]
    a_cols = np.shape(a)[2]
    b_cols = np.shape(b)[2]
    b_rows = np.shape(b)[1]
    c = np.zeros([np.shape(a)[0],a_rows,b_cols])
    ls = 256
    a = a.flatten()
    b = b.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        int z = (gidx0 / {a_rows}) / {b_cols};
        int x = (gidx0 / {a_rows}) % {b_cols};
        int y = gidx0 % {a_rows};
        float total = 0;
        for(int k = 0; k < {b_rows}; k++) {{
            total += a[y*{b_rows} + k + z*{a_rows}*{a_cols}] * b[x + k*{b_cols} + z*{b_rows}*{b_cols}]; 
        }}
        res[y*{b_cols} + x + z*{b_cols}*{a_rows}] = total;
    }}
    """).build()
    g = math.ceil((np.shape(a)[0]*b_cols*a_rows / ls) * ls)
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c
    

def matmul_t_3d_b(a,b):
    a_rows = np.shape(a)[1]
    a_cols = np.shape(a)[2]
    b_cols = np.shape(b)[2]
    b_rows = np.shape(b)[1]
    c = np.zeros([np.shape(a)[0],a_rows,b_cols])
    
    a = a.flatten()
    b = b.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int x = get_global_id(0);
        for(int z = 0; z < {np.shape(a)[0]}; z++) {{
            if(x < {b_cols}) {{
                for(int y = 0; y < {a_rows}; y++) {{
                    float total = 0;
                    for(int k = 0; k < {b_rows}; k++) {{
                        total += a[y*{b_rows} + k + z*{a_rows}*{a_cols}] * b[x + k*{b_cols} + z*{b_rows}*{b_cols}]; 
                    }}
                    res[y*{b_cols} + x + z*{b_cols}*{a_rows}] = total;
                }}  
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(b_cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_3d_c(a,b):
    a_rows = np.shape(a)[1]
    a_cols = np.shape(a)[2]
    n = np.shape(a)[0]
    c = np.zeros([n,a_rows,a_rows])
    ls = 256
    g = a_rows*a_rows*n
    g = math.ceil(g / ls) * ls
    a = a.flatten()
    b = b.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)


    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        if(gidx0 < {n}*{a_rows}*{a_rows}) {{
            int x = (gidx0 / {a_rows}) % {a_rows};
            int z = gidx0 / ({a_rows}*{a_rows}); 
            int y = gidx0 % {a_rows};
            float total = 0;
            for(int k = 0; k < {a_cols}; k++) {{
                total += a[y*{a_cols} + k + z*{a_rows}*{a_cols}] * b[x*{a_cols} + k + z*{a_cols}*{a_rows}]; 
            }}
            res[y*{a_rows} + x + z*{a_rows}*{a_rows}] = total / 8; //sqrt 64 input shape xq
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,c_g) #todo this will break when g < ls, small prompt
    cl.enqueue_copy(queue, c, c_g)
    return c


def minus_sum_3d(a):
    x = np.shape(a)[0]
    num_tokens = np.shape(a)[1]
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    res = np.zeros(num_tokens*x).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    ls = 256
    prg = cl.Program(ctx, f"""
    __kernel void ms(
        __global float *a)
    {{
    int gidx0 = get_global_id(0);
    if(gidx0 < {num_tokens*num_tokens*x}) {{
        int x = (gidx0 / {num_tokens}) / {num_tokens};
        int y = (gidx0 / {num_tokens}) % {num_tokens};
        int z = gidx0 % {num_tokens};
        if(z > y) {{ //todo, this can probably be 2x faster
            a[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = -INFINITY;
        }}
        a[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = exp(a[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z]);
    }}
    }}

    __kernel void ms3(
        __global float *a, __global float *mx)
    {{
    int gidx0 = get_global_id(0);
    int x = gidx0 / {num_tokens};
    int y = gidx0 % {num_tokens};
        float m = 0;
        for(int z = 0; z < {num_tokens}; z++) {{
            m += a[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z];
        }}
        mx[x*{num_tokens} + y] = m;  
    }}

    __kernel void ms4(
        __global float *a, global const float *mx)
    {{
    int gidx0 = get_global_id(0);
    if(gidx0 < {num_tokens*num_tokens*x}) {{
        int x = (gidx0 / {num_tokens}) / {num_tokens};
        int y = (gidx0 / {num_tokens}) % {num_tokens};
        int z = gidx0 % {num_tokens};
        a[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] /= mx[x*{num_tokens} + y];
    }}
    }}
    """).build()
    knl = prg.ms
    g = x*num_tokens*num_tokens
    g = math.ceil(g / ls) * ls
    knl(queue, (g,1), (ls,1), a_g)
    g2 = x*num_tokens #todo, will break for larger inputs
    knl3 = prg.ms3
    knl3(queue, (g2,1), (g2,1), a_g,res_g)
    knl4 = prg.ms4
    knl4(queue, (g,1), (ls,1), a_g,res_g)
    cl.enqueue_copy(queue, a, a_g)
    return a
    

def matvec4(a,b):
    b_cols = np.shape(b)[1]
    b_rows = np.shape(b)[0]
    c = np.zeros(b_cols)
    
    a = a.flatten()
    b = b.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int x = get_global_id(0);
        if(x < {b_cols}) {{
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += a[k] * b[x*{b_cols} + k]; 
            }}
            res[x] = total;
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(b_cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def transpose(a):
    a_rows = np.shape(a)[1]
    a_cols = np.shape(a)[2]
    a = a.flatten()
    at = np.zeros_like(a)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    at_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=at)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global float *at)
    {{
        for(int i = 0; i < 12; i++) {{
            for(int j = 0; j < {a_rows}; j++) {{
                for(int k = 0; k < {a_cols}; k++) {{
                    at[i*{a_rows}*{a_cols} + j + k*{a_rows}] = a[i*{a_rows}*{a_cols} + j*{a_cols} + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (1,1), (1,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at

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

#12,15,64
#50257
#(12, 13, 64) (12, 64, 13)
'''
a = np.random.rand(12, 13, 64).astype(np.float32)
b = np.random.rand(12, 64, 13).astype(np.float32)

n1,t1 = time_it(matmul_t_3d_b,a,b,20)
n2,t2 = time_it(matmul_t_3d_c,a,b,20)
np.testing.assert_allclose(n1,n2,rtol=1e-5)
print(t1,t2)
'''

	

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

def kernel_0_b(x,weight,bias,n_tokens,retnp=False):
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
    if retnp:
       cl.enqueue_copy(queue, x, x_g)
       return x 
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

def matmul_t_d(a,b,bias_g,h):
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
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b,__global const float *bias, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        int x = gidx0 / {a_rows};
        int y = gidx0 % {a_rows};
        float total = 0;
        for(int k = 0; k < {b_rows}; k++) {{
            total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
        }}
        res[y*{b_cols} + x] += total + bias[x];

    }}
    """).build()
    g = math.ceil((b_cols*a_rows / ls)*ls)
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,bias_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_d2(a,b,bias_g):
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
        __global const float *a, __global const float *b,__global const float *bias, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        int x = gidx0 / {a_rows};
        int y = gidx0 % {a_rows};
        float total = 0;
        for(int k = 0; k < {b_rows}; k++) {{
            total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
        }}
        res[y*{b_cols} + x] = 0.5 * (total + bias[x])\
            * (1 + tanh((total + bias[x]) * 0.7978845608\
            * (1 + 0.044715 * pow((total + bias[x]),2))));
    }}
    """).build()
    g = math.ceil((b_cols*a_rows / ls)*ls)
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,bias_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_e(a,b,bias_g,n_tokens,h):
    a_rows = n_tokens
    b_rows = 768
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
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b,__global const float *bias, __global float *res)
    {{
        int gidx0 = get_global_id(0);
        int x = gidx0 / {a_rows};
        int y = gidx0 % {a_rows};
        float total = 0;
        for(int k = 0; k < {b_rows}; k++) {{
            total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
        }}
        res[y*{b_rows} + x] += total + bias[x];
    }}
    """).build()
    g = math.ceil((b_rows*a_rows / ls)*ls)
    knl = prg.matmul
    knl(queue, (g,1), (ls,1), a_g, b_g,bias_g,h_g)
    cl.enqueue_copy(queue, h, h_g)
    return h

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
    # (12,13,64) -? (13,12,64)
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
        int gidx0 = get_global_id(0);
        int i = (gidx0 / 64) / {a_rows};
        int j = (gidx0 / 64) % {a_rows};
        int k = gidx0 % 64;
        at[i*64 + j*12*64 + k] = a[i*{a_rows}*64 + j*64 + k];
    }}
    """).build()
    knl = prg.matmul
    g = a_rows*12*64
    ls = 256
    g = math.ceil(g / ls)*ls
    knl(queue, (g,1), (ls,1), a_g, at_g)
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

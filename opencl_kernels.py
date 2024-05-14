import numpy as np
import pyopencl as cl
import time
import math

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

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
        res[gidx0] = a[{a_s*768} + gidx0] + b[gidx0 + {b_s}*768];   
    }}
    """).build()
    knl = prg.add
    knl(queue, (768,1), (256,1), a_g, b_g,res_g) #todo check shape
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

len = 768
loop_size = int(len / 256)
len_short = 768

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

def kernel_2(a,c,d,e,f,g,keys,start_pos): #g = size
    size = np.shape(a)[0]
    ls = 256
    seg_e = int(np.shape(e)[1] / ls)
    rows_e = np.shape(e)[0]
    xq = f[0:g]
    xk = f[g:2*g]
    xv = f[2*g:]
    seg = int(size / ls) #todo
    keys_shape = np.shape(keys)[1] * np.shape(keys)[2]
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    e_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e)
    f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
    xk_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xk)
    xv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xv)
    keys_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d, __global const float *e,
        __global const float *f, __global float *xq, __global float *xk, __global float *xv, __global float *keys)
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
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg_e}; i++) {{
            float total = 0;
            for(int k = 0; k < {rows_e}; k++) {{
                total += a[k] * e[(lidx0*{seg_e} + i)*{rows_e} + k]; 
            }}
        if((lidx0*{seg_e} + i) < {g}) {{xq[lidx0*{seg_e} + i] += total;}}
        if((lidx0*{seg_e} + i) >= {g} && (lidx0*{seg_e} + i) < {2*g}) {{
            xk[lidx0*{seg_e} + i - {g}] += total; //TODO don't need this?
            keys[{start_pos}*{keys_shape} + lidx0*{seg_e} + i - {g}] = xk[lidx0*{seg_e} + i - {g}];
        }}
        if((lidx0*{seg_e} + i) >= {2*g}) {{xv[lidx0*{seg_e} + i - {2*g}] += total;}}
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1), a_g, c_g, d_g, e_g,f_g,xq_g,xk_g,xv_g,keys_g)
    cl.enqueue_copy(queue, xq, xq_g)
    cl.enqueue_copy(queue, xv, xv_g)
    cl.enqueue_copy(queue, keys, keys_g)
    return xq,xv,keys

def kernel_1(a,c,d,e,f,g,h):
    rows = 768
    cols = 3072
    size = np.shape(a)[0]
    ls = 256
    seg = int(size / ls)
    f = f.flatten()
    f = np.float32(f)
    g_rows = np.shape(g)[0]
    g_cols = np.shape(g)[1]
    g = g.flatten()
    g = np.float32(g)
    h = h.flatten()
    h = np.float32(h)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    e_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e)
    f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    g_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    l_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.copy(a))
    prg = cl.Program(ctx, f"""
    __kernel void knl(
        __global float *a, __global const float *c, __global const float *d,
        __global const float *e, __global float *f, __global const float *g,
        __global float *h, __global const float *l)
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
        barrier(CLK_LOCAL_MEM_FENCE);    
        for(int i = 0; i < {int(cols / ls)}; i++) {{
            for(int j = 0; j < {rows}; j++) {{
                f[i + lidx0*{int(cols / ls)}] += a[j] * e[(i + lidx0*{int(cols / ls)})*{rows} + j];
            }}
        }} 
        barrier(CLK_LOCAL_MEM_FENCE);  
        for(int i = 0; i < {int(cols / ls)}; i++) {{
            f[i + lidx0*{int(cols / ls)}] = 0.5 * f[i + lidx0*{int(cols / ls)}]\
            * (1 + tanh(f[i + lidx0*{int(cols / ls)}] * 0.7978845608\
            * (1 + 0.044715 * pow(f[i + lidx0*{int(cols / ls)}],2))));
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  //TODO this is probably slower than it should be
        for(int i = 0; i < {int(np.shape(h)[0] / ls)}; i++) {{
            for(int j = 0; j < {g_rows}; j++) {{
                h[lidx0 + i*{ls}] += f[j] * g[lidx0 + i*{ls} + j*{g_cols}];
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {int(np.shape(h)[0] / ls)}; i++) {{
            h[i + lidx0*{int(np.shape(h)[0] / ls)}] += l[i + lidx0*{int(np.shape(h)[0] / ls)}];
        }}
    }}
    """).build()
    knl = prg.knl
    knl(queue, (ls,1), (ls,1), a_g, c_g, d_g, e_g,f_g,\
    g_g,h_g,l_g)
    cl.enqueue_copy(queue, h, h_g)
    return h

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

def kernel_3(a,keys,values):
    s = np.shape(keys)[2]
    values = values.flatten()
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    keys = keys.flatten() #todo, shouldnt be needed
    keys_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    valuest = np.zeros_like(values).astype(np.float32)
    valuest_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=valuest)
    xq = np.zeros(12*s).astype(np.float32)
    xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
    seg2 = math.ceil((np.shape(values)[0] / 64) / 256)
    res = np.zeros([12,1,64]).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))
    ls = 256
    seg = int((12*64) / ls)
    prg = cl.Program(ctx, f"""
    __kernel void k(
        __global const float *a, __global const float *keys, __global const float *values,
        __global float *valuest, __global float *xq, __global float *res)
    {{
    int lidx0 = get_local_id(0);
    for(int z = 0; z < 12; z++) {{
        for(int k = 0; k < {s*s}; k+={ls}) {{
            int x = (k + z*{s*s} + lidx0) % {s};
            int y = (k + z*{s*s} + lidx0) / {s};
            float acc0 = 0.0f;
            for(int i = 0; i < 64; i++) {{
                acc0 += a[i + 64*y] * keys[z*{s*s} + x + i*{s} + y*{s}*64];
            }}                  
            xq[x + y*{s}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx0 < 12){{
    float m = -INFINITY;
    for(int i = 0; i < {s}; i++) {{
        float val = xq[i + lidx0*{s}];
        m = max(m,val);
    }}
    for(int i = 0; i < {s}; i++) {{
        xq[i + lidx0*{s}] = exp(xq[i + lidx0*{s}] - m);
    }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx0 < 12) {{
    float t = 0;
    for(int i = 0; i < {s}; i++) {{
        float val = xq[i + lidx0*{s}];
        t = t+val;
    }}
    for(int i = 0; i < {s}; i++) {{
        xq[i + lidx0*{s}] = xq[i + lidx0*{s}] / t;
    }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < {seg2}; i++) {{
        int y = (lidx0*{seg2} + i) / 12;
        int x = (lidx0*{seg2} + i) % 12;
        for(int k = 0; k < 64; k++) {{
        if((y*12*64 + x*64 + k) < {np.shape(values)[0]}) {{
            valuest[x*64*{s} + y*64 + k] = values[y*12*64 + x*64 + k];
        }}
    }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int g = 0; g < {seg}; g++) {{
        int y = (g + lidx0*{seg}) / 64;
        int x = (g + lidx0*{seg}) % 64;
        float acc0 = 0;
        for(int i = 0; i < {s}; i++) {{
            acc0 += xq[i + {s}*y] * valuest[i*64 + x + y*{s}*64];
        }}
        res[x + y*64] = acc0;
    }}
    }}
    """).build()
    knl = prg.k
    knl(queue, (ls,1), (ls,1), a_g, keys_g, values_g, valuest_g, xq_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

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
                    at[x*64*{s} + y*64 + k] = a[y*12*64 + x*64 + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (256,1), (256,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at.reshape(12,s,64)

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

def matvec_b(a,b,c):
    ls = 256
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    a = a.flatten()
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros([768])
    d = np.float32(d)
    len_mv = np.shape(a)[0]
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b, __global float *c)
    {{
        int lidx0 = get_local_id(0);
        for(int i = 0; i < {int(len_mv / ls)}; i++) {{
            float acc = 0;
            for(int x = 0; x < {len_mv}; x++) {{
                acc += a[x] * b[x*{len_mv} + lidx0*{int(len_mv / ls)} + i];
            }}
            c[lidx0*{int(len_mv / ls)} + i] = acc + c[lidx0*{int(len_mv / ls)} + i];
        }}
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (256,1), (256,1), a_g, b_g,c_g)
    cl.enqueue_copy(queue, d, c_g)
    cl.enqueue_copy(queue, c, c_g)
    np.testing.assert_allclose(d,c,rtol=1e-5)
    return c

def matvec2(a,b,c): #pass bias in instead of adding to zero, todo for other kernels
    rows = np.shape(b)[0]
    cols = np.shape(b)[1]
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = c.flatten()
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int j = 0; j < {rows}; j++) {{
            res[gidx0] += a[j] * b[gidx0 + j*{cols}];
        }}
    }}
    """).build()
    knl = prg.matvec
    gidx = math.ceil(cols / 16) * 16
    knl(queue, (gidx,1), (16,1), a_g, b_g,c_g)
    cl.enqueue_copy(queue, c, c_g)
    return c

def matvec2_b(a,b,c): #pass bias in instead of adding to zero, todo for other kernels
    ls = 256
    rows = np.shape(b)[0]
    cols = np.shape(b)[1]
    print("rows cols =",rows,cols)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = c.flatten()
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b , __global float *res)
    {{
        int lidx0 = get_local_id(0);
        for(int i = 0; i < 3; i++) {{
            for(int j = 0; j < 3072; j++) {{
                res[lidx0 + i*256] += a[j] * b[lidx0 + i*256 + j*768];
            }}
        }}
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (256,1), (256,1), a_g, b_g,c_g)
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t(a,b):
    a_rows = np.shape(a)[0]
    b_cols = np.shape(b)[1]
    b_rows = np.shape(b)[0]
    c = np.zeros([a_rows,b_cols])
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
            for(int y = 0; y < {a_rows}; y++) {{
                float total = 0;
                for(int k = 0; k < {b_rows}; k++) {{
                    total += a[y*{b_rows} + k] * b[x*{b_rows} + k]; 
                }}
                res[y*{b_cols} + x] = total;
            }}  
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(b_cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
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



def matmul_t_b(a,b):
    ls = 256
    seg = int(np.shape(b)[1] / ls)
    rows = np.shape(b)[0]
    c = np.zeros([np.shape(b)[1]]).astype(np.float32)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int lidx0 = get_local_id(0);
        for(int i = 0; i < {seg}; i++) {{
            float total = 0;
            for(int k = 0; k < {rows}; k++) {{
                total += a[k] * b[(lidx0*{seg} + i)*{rows} + k]; 
            }}
        res[lidx0*{seg} + i] = total;
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (256,1), (256,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_t_3d(a,b):
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
    s = np.shape(a)[0]    
    a = a.flatten()
    at = np.zeros_like(a)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    at_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=at)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global float *at)
    {{
        for(int i = 0; i < {s}; i++) {{
            for(int j = 0; j < 12; j++) {{
                for(int k = 0; k < 64; k++) {{
                    at[j*64*{s} + i*64 + k] = a[i*12*64 + j*64 + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (1,1), (1,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at.reshape(12,s,64)

def time_it(func,a,i=100):
    f = None
    total_time = 0
    for _ in range(i):
        st = time.perf_counter()
        ret = func(a)
        t = time.perf_counter() - st
        total_time += t
        if f is None or t < f:
            f = t
    return ret,f

#12,15,64
'''
i = 15
for i in range(12,130):
    a = np.random.rand(i,12,64).astype(np.float32)
    b_np = a.transpose(1,0,2)
    bf,tf = time_it(transpose_f,a,100)
    b,t = time_it(transpose,a,100)
    np.testing.assert_allclose(b,b_np,rtol=1e-5)
    np.testing.assert_allclose(bf,b_np,rtol=1e-5)
    print(i,"times =\t",t,"\t",tf)
'''
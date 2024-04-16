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
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void sum(
        __global float *data0)
    {{
    float avg = 0;
    __attribute__ ((aligned (16))) __local float temp[256];
    int lidx0 = get_local_id(0); /* 256 */
    int gidx0 = get_global_id(0); /* 256 */
    float acc0 = 0.0f;
    for (int ridx0 = 0; ridx0 < {loop_size}; ridx0++) {{
        float val0 = data0[(lidx0*{loop_size})+ridx0];
        acc0 = (val0+acc0);
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
    for(int i = 0; i < {loop_size}; i++) {{
        data0[lidx0*{loop_size} + i] = data0[lidx0*{loop_size} + i] - avg;
    }}
    }}
    """).build()
    knl = prg.sum
    knl(queue, (256,1), (256,1), a_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    return a

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

def minus_mean(a):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void minus_mean(
        __global float *a)
    {{
    float avg = 0;
    for(int i = 0; i < {len_short}; i++) {{
        avg += a[i];
    }}
    avg = avg / {len_short};
    int gidx0 = get_global_id(0);
    a[gidx0] = a[gidx0] - avg;
    }}
    """).build()
    knl = prg.minus_mean

    knl(queue, (768,1), (256,1), a_g) #has to be multiple of 256
    cl.enqueue_copy(queue, a, a_g)
    return a

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

def madd(a,b,c):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros((1,768*3)).astype(np.float32)
    d = np.float32(d)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)

    prg = cl.Program(ctx, f"""
    __kernel void add(const __global float* data1, const __global float* data2, const __global float* data3, __global float* data0) {{
  __attribute__ ((aligned (16))) __local float temp[128];
    int gidx0 = get_group_id(0); /* 144 */
    int lidx1 = get_local_id(1); /* 4 */
    int lidx2 = get_local_id(0); /* 8 */
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int alu0 = ((gidx0*16)+lidx1);
    int alu1 = (lidx1*32);
    float val0 = data3[alu0];
    int alu2 = (alu0+4);
    float val1 = data3[alu2];
    int alu3 = (alu0+8);
    float val2 = data3[alu3];
    int alu4 = (alu0+12);
    float val3 = data3[alu4];
    for (int ridx0 = 0; ridx0 < 96; ridx0++) {{
        float val4 = data1[lidx2+(ridx0*8)];
        int alu5 = (alu0+(lidx2*2304)+(ridx0*18432));
        float val5 = data2[alu5];
        float val6 = data2[alu5+4];
        float val7 = data2[alu5+8];
        float val8 = data2[alu5+12];
        acc0 = mad(val4,val5,acc0);
        acc1 = mad(val4,val6,acc1);
        acc2 = mad(val4,val7,acc2);
        acc3 = mad(val4,val8,acc3);
    }}
    *((__local float4*)(temp+alu1+(lidx2*4))) = (float4)(float4)(acc0,acc1,acc2,acc3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lidx2<1)) {{
        float4 acc4 = (float4)(0.0f,0.0f,0.0f,0.0f);
        for (int ridx1 = 0; ridx1 < 8; ridx1++) {{
        float4 val9 = (float4)(*((__local float4*)(temp+alu1+(ridx1*4))));
        (acc4).x = ((val9).x+(acc4).x);
        (acc4).y = ((val9).y+(acc4).y);
        (acc4).z = ((val9).z+(acc4).z);
        (acc4).w = ((val9).w+(acc4).w);
        }}
        data0[alu0] = ((float)((acc4).x)+val0);
        data0[alu2] = ((float)((acc4).y)+val1);
        data0[alu3] = ((float)((acc4).z)+val2);
        data0[alu4] = ((float)((acc4).w)+val3);
    }}
    }}
    """).build()
    knl = prg.add
    knl(queue, (1152,4,1), (8,4,1), a_g, b_g, c_g ,d_g) #todo check shape
    cl.enqueue_copy(queue, d, d_g)
    return d

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

def matvec(a,b,c):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros([1,1,768])
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

def matmul(a,b):
    cols = np.shape(b)[1]
    rows = np.shape(b)[0]
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([13,cols])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int x = get_global_id(0);
        if(x < {cols}) {{
            for(int y = 0; y < 13; y++) {{
                float total = 0;
                for(int k = 0; k < {rows}; k++) {{
                    total += a[y*{rows} + k] * b[x + k*{cols}]; 
                }}
                res[y*{cols} + x] = total;
            }}  
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmulcr(a,b): #column-row weight (b)
    cols = np.shape(b)[0]
    rows = np.shape(b)[1]
    a_rows = np.shape(a)[0]
    print("rory a_rows =",a_rows)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([a_rows,rows])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int x = get_global_id(0);
        if(x < {cols}) {{
            for(int y = 0; y < {a_rows}; y++) {{
                float total = 0;
                for(int k = 0; k < {rows}; k++) {{
                    total += a[y*{cols} + k] * b[x*{rows} + k]; 
                }}
                res[y*{cols} + x] = total;
            }}  
        }}
    }}
    """).build()
    knl = prg.matmul
    group_size = math.ceil(cols / 16) * 16
    knl(queue, (group_size,1), (16,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmulb(a,b):
    cols = np.shape(b)[1]
    rows = np.shape(b)[0]
    print("B[0][1] =",b[0][1],b[1][0])
    #b = b.transpose()
    print("B[0][1] =",b[0][1],b[1][0])
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([13,cols])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        res[1] = b[1];
    }}
    """).build()
    knl = prg.matmul
    print("rory rows cols =",rows,cols)
    group_size = math.ceil(cols / 16) * 16
    knl(queue, (1,1), (1,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

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
        #print(f)
    return ret,f

'''
for i in range(1):
    a = np.random.rand(13,768).astype(np.float32)
    b = np.random.rand(768,768).astype(np.float32)
    c_np = np.matmul(a,b)
    c = matmul(a,b)
    np.testing.assert_allclose(c,c_np,rtol=1e-5)

    a = np.random.rand(13,768).astype(np.float32) 
    b = np.random.rand(768,3072).astype(np.float32)
    c_np = np.matmul(a,b)
    c = matmul(a,b)
    np.testing.assert_allclose(c,c_np,rtol=1e-5)

    a = np.random.rand(13,3072).astype(np.float32) 
    b = np.random.rand(3072,768).astype(np.float32)
    c_np = np.matmul(a,b)
    c = matmul(a,b)
    np.testing.assert_allclose(c,c_np,rtol=1e-5)

    a = np.random.rand(13,3072).astype(np.float32) 
    b = np.random.rand(3072,3072).astype(np.float32)
    c_np = np.matmul(a,b)
    c = matmul(a,b)
    np.testing.assert_allclose(c,c_np,rtol=1e-5)

    a = np.random.rand(13,768).astype(np.float32) 
    b = np.random.rand(768,50257).astype(np.float32)
    c_np = np.matmul(a,b)
    c = matmul(a,b)
    np.testing.assert_allclose(c,c_np,rtol=1e-5)
'''
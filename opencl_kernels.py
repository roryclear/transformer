import numpy as np
import pyopencl as cl
import time

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

len = 768000
loop_size = int(len / 256)
len_short = 768
def minus_mean_large(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void sum(
        __global float *data0, const __global float *data1)
    {{
    float avg = 0;
    __attribute__ ((aligned (16))) __local float temp[256];
    int lidx0 = get_local_id(0); /* 256 */
    int gidx0 = get_global_id(0); /* 256 */
    float acc0 = 0.0f;
    for (int ridx0 = 0; ridx0 < {loop_size}; ridx0++) {{
        float val0 = data1[(lidx0*{loop_size})+ridx0];
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

    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    knl(queue, (256,1), (256,1), a_g, res_g) #rory to test large stuff
    cl.enqueue_copy(queue, a, a_g)
    return a

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
'''
a = np.random.rand(len).astype(np.float32)
b = np.copy(a)
c = minus_mean_large(a,a)
cnp = b - np.mean(b)
np.testing.assert_allclose(c,cnp,atol=1e-6)
a = np.random.rand(len).astype(np.float32)
b = np.random.rand(len).astype(np.float32)

st = time.perf_counter()
for i in range(100):
    c_m = minus_mean(a,b) #same but multi
t = time.perf_counter() - st
print("rory time taken multi =",t)

st = time.perf_counter()
for i in range(100):
    c = sum(a,b)
t = time.perf_counter() - st
print("rory time taken =",t)

c_np = a - np.mean(a)
#print("rory a and c",c,c_np)
np.testing.assert_allclose(c,c_np,atol=1e-4)
np.testing.assert_allclose(c,c_m,atol=1e-4)
minus mean stuff

a = np.random.rand(1,1,768).astype(np.float32)
b = np.random.rand(1,1,768).astype(np.float32)

st = time.perf_counter()
for i in range(1000):
    c = add(a,b)
clt = time.perf_counter() - st
c = c.reshape(1,1,768)
st = time.perf_counter()
for i in range(100000):
    c_np = a+b
npt = time.perf_counter() - st
print("rory shapes =",np.shape(c),np.shape(a+b))
np.testing.assert_allclose(c,(a+b),rtol=0.001)
print("passed",clt,npt)
'''
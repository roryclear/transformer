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

len = 768
def sum(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void sum(
        __global float *a, __global float *res)
    {{
    float avg = 0;
    for(int i = 0; i < {len}; i++) {{
        avg += a[i];
    }}
    avg = avg / {len};
    for(int i = 0; i < {len}; i++) {{
        res[i] = a[i] - avg;
    }}
    }}
    """).build()
    knl = prg.sum

    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    knl(queue, (1,1), (1,1), a_g, res_g)
    cl.enqueue_copy(queue, b, res_g)
    return b

def minus_mean(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    prg = cl.Program(ctx, f"""
    __kernel void sum(
        __global float *a, __global float *res)
    {{
    float avg = 0;
    for(int i = 0; i < {len}; i++) {{
        avg += a[i];
    }}
    avg = avg / {len};
    int gidx0 = get_global_id(0);
    res[gidx0] = a[gidx0] - avg;
    }}
    """).build()
    knl = prg.sum

    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    knl(queue, (len,1), (256,1), a_g, res_g) #has to be multiple of 256
    cl.enqueue_copy(queue, b, res_g)
    return b

'''
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
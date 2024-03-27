import numpy as np
import pyopencl as cl
import time

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def add(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    res_np = np.zeros(np.shape(a)).astype(np.float32).flatten()
    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void add(
        __global const float *a, __global float *res)
    {{
    int gidx0 = get_global_id(0);
        res[gidx0] += a[gidx0];   
    }}
    """).build()
    knl = prg.add
    knl(queue, (768,1), (256,1), a_g, res_g) #todo check shape
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

'''
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
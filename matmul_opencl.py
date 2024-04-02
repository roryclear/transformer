import numpy as np
import pyopencl as cl
import time
np.random.seed(420)
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def matmul(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([1,2304])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int lidx0 = get_local_id(0);
        for(int j = 0; j < 768; j++) {{
            for(int i = 0; i < 2304; i++) {{
                res[i] += a[j] * b[i + j*2304];
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (1,1), (1,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul_small(a,b): #start small make bigger idk
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([1,12])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
        int gidx0 = get_group_id(0); /* 3 */
        int lidx1 = get_local_id(0); /* 4 */
        float acc0 = 0.0f;
        float4 val0 = (float4)(*((__global float4*)(a+0)));
        int alu0 = ((gidx0*4)+lidx1);
        float val1 = b[alu0];
        float val2 = b[alu0+12];
        float val3 = b[alu0+24];
        float val4 = b[alu0+36];
        res[alu0] = mad((val0).w,val4,mad((val0).z,val3,mad((val0).y,val2,mad((val0).x,val1,acc0))));
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (12,1), (4,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

'''
a = np.random.rand(1,768).astype(np.float32)
b = np.random.rand(768,2304).astype(np.float32)
c = np.zeros([1,2304])
c = np.float32(c)

st = time.perf_counter()
for _ in range(10):
    c = matmul(a,b)
t = time.perf_counter() - st
print("time taken =",t)
st = time.perf_counter()
for _ in range(10):
    d = np.matmul(a,b)
t = time.perf_counter() - st
print("time taken np =",t)

np.testing.assert_allclose(c,d,rtol=1e-5)
'''

a = np.random.rand(1,4).astype(np.float32)
b = np.random.rand(4,12).astype(np.float32)
c = np.zeros([1,12])
c = np.float32(c)

st = time.perf_counter()
for _ in range(10):
    c = matmul_small(a,b)
t = time.perf_counter() - st
print("time taken =",t)
st = time.perf_counter()
for _ in range(10):
    d = np.matmul(a,b)
t = time.perf_counter() - st
print("time taken np =",t)

np.testing.assert_allclose(c,d,rtol=1e-5)


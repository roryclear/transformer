import numpy as np
import pyopencl as cl
import time
np.random.seed(420)
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
print(my_gpu_devices[0])
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def matmul(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([1,112])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
    for(int x = 0; x < 1; x++) {{
        for(int i = 0; i < 112; i++) {{
            for(int j = 0; j < 64; j++) {{
                res[i] += a[j] * b[i + 112*j];    
            }}             
        }}
    }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (1,1), (1,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c


def matmul2(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([1,4])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *a, __global const float *b, __global float *res)
    {{
    for(int y = 0; y < 10000; y++) {{
        int x = get_local_id(0) + get_group_id(0) * 16;
        float acc0 = 0.0f;
        
        float4 vals_a[16];
        for(int i = 0; i < 16; i++) {{
            vals_a[i] = (float4)(*((__global float4*)(a+i*4)));
        }}
                     
        float vals_b[64];
        for(int i = 0; i < 64; i++) {{
            vals_b[i] = b[x+i*4];
        }}
        
        for(int i = 0; i < 16; i++) {{
            acc0 = mad((vals_a[i]).x,vals_b[i*4],acc0);
            acc0 = mad((vals_a[i]).y,vals_b[i*4 + 1],acc0);
            acc0 = mad((vals_a[i]).z,vals_b[i*4 + 2],acc0);
            acc0 = mad((vals_a[i]).w,vals_b[i*4 + 3],acc0);
        }}
        res[x] = acc0;
    }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (64,1), (64,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul2_tg(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([1,4])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *data1, __global const float *data2, __global float *data0)
    {{
    __attribute__ ((aligned (16))) __local float temp[16];
    int gidx0 = get_group_id(0); /* 4 */
    int lidx1 = get_local_id(0); /* 16 */
    float acc0 = 0.0f;
    for (int ridx0 = 0; ridx0 < 4; ridx0++) {{
        float val0 = data1[(lidx1*4)+ridx0];
        float val1 = data2[gidx0+(lidx1*16)+(ridx0*4)];
        acc0 = mad(val0,val1,acc0);
    }}
    temp[lidx1] = acc0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lidx1<1)) {{
        float acc1 = 0.0f;
        for (int ridx1 = 0; ridx1 < 16; ridx1++) {{
        float val2 = temp[ridx1];
        acc1 = (val2+acc1);
        }}
        data0[gidx0] = acc1;
    }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (64,1), (16,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

def time_it(func,a,b,s,i=1):
    f = None
    total_time = 0
    for _ in range(i):
        st = time.perf_counter()
        ret = func(a,b)
        t = time.perf_counter() - st
        total_time += t
        if f is None or t < f:
            f = t
    print("time taken\t",s,"\t",f,"\t",total_time)
    return ret

####SMALL MATMUL...MAKE BIGGER###
a = np.random.rand(1,64).astype(np.float32)
b = np.random.rand(64,4).astype(np.float32)
c_np = np.matmul(a,b)
c2 = time_it(matmul2_tg,a,b,"tg",100)
c = time_it(matmul2,a,b,"mine",100)
print(c,c_np)
np.testing.assert_allclose(c,c_np,rtol=1e-5)
np.testing.assert_allclose(c2,c_np,rtol=1e-5)


a = np.random.rand(1,64).astype(np.float32)
b = np.random.rand(64,112).astype(np.float32)
#a.fill(1)
#b.fill(1)
c = np.zeros([1,112])
c = np.float32(c)

st = time.perf_counter()
c = matmul(a,b)
t = time.perf_counter() - st
print("time taken:",t)

c_np = np.matmul(a,b)
np.testing.assert_allclose(c,c_np,rtol=1e-6)


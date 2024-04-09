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

#try using two big tiles????
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
    int x = get_local_id(0);
    //for(int x = 0; x < 4; x++) {{
        float acc0 = 0.0f;
        float4 val0 = (float4)(*((__global float4*)(a+0)));
        float4 val9 = (float4)(*((__global float4*)(a+4)));
        float val1 = b[x];
        float val2 = b[x+4];
        float val3 = b[x+8];
        float val4 = b[x+12];
        float val5 = b[x+16];
        float val6 = b[x+20];
        float val7 = b[x+24];
        float val8 = b[x+28];
        
        acc0 = mad((val0).x,val1,acc0);
        acc0 = mad((val0).y,val2,acc0);
        acc0 = mad((val0).z,val3,acc0);
        acc0 = mad((val0).w,val4,acc0);
                     
        acc0 = mad((val9).x,val5,acc0);
        acc0 = mad((val9).y,val6,acc0);
        acc0 = mad((val9).z,val7,acc0);
        acc0 = mad((val9).w,val8,acc0);
        res[x] = acc0;
    //}}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (4,1), (4,1), a_g, b_g,c_g) #todo check shape
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
    int lidx0 = get_local_id(0); /* 4 */
    float acc0 = 0.0f;
    float val0 = data1[0];
    float val1 = data1[1];
    float val2 = data1[2];
    float val3 = data1[3];
    float val4 = data1[4];
    float val5 = data1[5];
    float val6 = data1[6];
    float val7 = data1[7];
    float val8 = data2[lidx0];
    float val9 = data2[lidx0+4];
    float val10 = data2[lidx0+8];
    float val11 = data2[lidx0+12];
    float val12 = data2[lidx0+16];
    float val13 = data2[lidx0+20];
    float val14 = data2[lidx0+24];
    float val15 = data2[lidx0+28];
    data0[lidx0] = mad(val7,val15,mad(val6,val14,mad(val5,val13,mad(val4,val12,mad(val3,val11,mad(val2,val10,mad(val1,val9,mad(val0,val8,acc0))))))));
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (4,1), (4,1), a_g, b_g,c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

def time_it(func,a,b,s,i=1):
    f = None
    for _ in range(i):
        st = time.perf_counter()
        ret = func(a,b)
        t = time.perf_counter() - st
        if f is None or t < f:
            f = t
    print("time taken\t",s,"\t",f)
    return ret

####SMALL MATMUL...MAKE BIGGER###
a = np.random.rand(1,8).astype(np.float32)
b = np.random.rand(8,4).astype(np.float32)
c_np = np.matmul(a,b)
c2 = time_it(matmul2_tg,a,b,"tg",20)
c = time_it(matmul2,a,b,"mine",20)
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
print(c)
np.testing.assert_allclose(c,c_np,rtol=1e-6)


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

def madd(a,b):
    #a = np.float32(a)
    #b = np.float32(b)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros((1,768*3)).astype(np.float32)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    prg = cl.Program(ctx, f"""
    __kernel void add(const __global float* data1, const __global float* data2, __global float* data0) {{
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
        for (int ridx0 = 0; ridx0 < 96; ridx0++) {{
            float val0 = data1[lidx2+(ridx0*8)];
            int alu2 = (alu0+(lidx2*2304)+(ridx0*18432));
            float val1 = data2[alu2];
            float val2 = data2[alu2+4];
            float val3 = data2[alu2+8];
            float val4 = data2[alu2+12];
            acc0 = mad(val0,val1,acc0);
            acc1 = mad(val0,val2,acc1);
            acc2 = mad(val0,val3,acc2);
            acc3 = mad(val0,val4,acc3);
        }}
        *((__local float4*)(temp+alu1+(lidx2*4))) = (float4)(float4)(acc0,acc1,acc2,acc3);
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((lidx2<1)) {{
            float4 acc4 = (float4)(0.0f,0.0f,0.0f,0.0f);
            for (int ridx1 = 0; ridx1 < 8; ridx1++) {{
            float4 val5 = (float4)(*((__local float4*)(temp+alu1+(ridx1*4))));
            (acc4).x = ((val5).x+(acc4).x);
            (acc4).y = ((val5).y+(acc4).y);
            (acc4).z = ((val5).z+(acc4).z);
            (acc4).w = ((val5).w+(acc4).w);
            }}
            data0[alu0] = (acc4).x;
            data0[alu0+4] = (acc4).y;
            data0[alu0+8] = (acc4).z;
            data0[alu0+12] = (acc4).w;
        }}
    }}
    """).build()
    knl = prg.add
    knl(queue, (1152,4,1), (8,4,1), a_g, b_g, c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    return c

def madd2(a,b):
    a = np.float32(a)
    b = np.float32(b)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros((1,768*3),dtype=np.float32)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    
    prg = cl.Program(ctx, f"""
    __kernel void add(const __global float* data1, __global float* data2, __global float* data0) {{
        data0[0] = data2[0];
        data0[1] = data2[768];
    }}
    """).build()
    knl = prg.add
    print("rory BEFORE c[1] =",c[0][1],type(c[0][1]),"should be",b[0][1],type(b[0][1]))
    knl(queue, (1,1,1), (1,1,1), a_g, b_g, c_g) #todo check shape
    cl.enqueue_copy(queue, c, c_g)
    print("rory c[1] =",c[0][1],type(c[0][1]),"should be",b[0][1],type(b[0][1]))
    return c

a = np.random.rand(1,768)
b = np.random.rand(768,768*3)
a,b = np.float32(a), np.float32(b)
c = madd(a,b)
print(c)
c2 = np.matmul(a,b)
print(c2)
print(np.shape(c),np.shape(c2))
#np.testing.assert_allclose(c,c2,rtol=1e-5)

###these kernel names make no sense atm
'''
a = np.random.rand(len).astype(np.float32)
b = np.copy(a)
st = time.perf_counter()
for i in range(10000):
    c = minus_mean(a)
t1 = time.perf_counter() - st
st = time.perf_counter()
for i in range(10000):
    c2 = minus_mean_multi(a)
t2 = time.perf_counter() - st
print("t1 =",t1)
print("t2 =",t2)
cnp = b - np.mean(b)
np.testing.assert_allclose(c,cnp,atol=1e-6)
np.testing.assert_allclose(c2,cnp,atol=1e-6)
'''
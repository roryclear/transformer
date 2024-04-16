import numpy as np
import pyopencl as cl
import time
import math
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


def matmul2(a,b,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
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
    res[x + k*{s}] = acc0;
    }}
    """).build()
    knl = prg.matmul
    local0 = min(256,s*64*12)
    group0 = math.ceil(12*112 / local0) * local0
    knl(queue, (group0,1), (local0,1), a_g, b_g,c_g) #shape of output !!!!!
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul2_tg(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([12,1,112])
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void matmul(
        __global const float *data1, __global const float *data2, __global float *data0)
    {{
    __attribute__ ((aligned (16))) __local float temp[128];
    int gidx0 = get_group_id(1); /* 12 */
    int gidx1 = get_group_id(0); /* 7 */
    int lidx2 = get_local_id(1); /* 4 */
    int lidx3 = get_local_id(0); /* 8 */
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int alu0 = (gidx1*16);
    int alu1 = (lidx2*32);
    int alu2 = ((gidx0*112)+alu0+lidx2);
    for (int ridx0 = 0; ridx0 < 8; ridx0++) {{
        float val0 = data1[(gidx0*64)+lidx3+(ridx0*8)];
        int alu3 = ((gidx0*7168)+alu0+lidx2+(lidx3*112)+(ridx0*896));
        float val1 = data2[alu3];
        float val2 = data2[alu3+4];
        float val3 = data2[alu3+8];
        float val4 = data2[alu3+12];
        acc0 = mad(val0,val1,acc0);
        acc1 = mad(val0,val2,acc1);
        acc2 = mad(val0,val3,acc2);
        acc3 = mad(val0,val4,acc3);
    }}
    *((__local float4*)(temp+alu1+(lidx3*4))) = (float4)(float4)(acc0,acc1,acc2,acc3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lidx3<1)) {{
        float4 acc4 = (float4)(0.0f,0.0f,0.0f,0.0f);
        for (int ridx1 = 0; ridx1 < 8; ridx1++) {{
        float4 val5 = (float4)(*((__local float4*)(temp+alu1+(ridx1*4))));
        (acc4).x = ((val5).x+(acc4).x);
        (acc4).y = ((val5).y+(acc4).y);
        (acc4).z = ((val5).z+(acc4).z);
        (acc4).w = ((val5).w+(acc4).w);
        }}
        data0[alu2] = (acc4).x;
        data0[alu2+4] = (acc4).y;
        data0[alu2+8] = (acc4).z;
        data0[alu2+12] = (acc4).w;
    }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (56,48), (8,4), a_g, b_g,c_g) #todo check shape 16*x and 16*1 (DONT CHANGE)
    cl.enqueue_copy(queue, c, c_g)
    return c

#for x = x - x.max() or something similar
def double_reduce(a):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c = np.zeros(1024)
    c = np.float32(c)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void red(
        __global const float *data1, __global float *data0)
    {{
    __attribute__ ((aligned (16))) __local float temp[256];
    __attribute__ ((aligned (16))) __local float m;
    int lidx0 = get_local_id(0); /* 256 */
    int gidx0 = get_group_id(0);
    float acc0 = -INFINITY;
    for (int ridx0 = 0; ridx0 < 4; ridx0++) {{
        float val0 = data1[(lidx0*4)+ridx0];
        float alu0 = max(val0,acc0);
        acc0 = alu0;
    }}
    temp[lidx0] = acc0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lidx0<1)) {{
        float acc1 = -INFINITY;
        for (int ridx1 = 0; ridx1 < 256; ridx1++) {{
        float val1 = temp[ridx1];
        float alu1 = max(val1,acc1);
        acc1 = alu1;
        }}
        m = acc1;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 val0 = (float4)(*((__global float4*)(data1+lidx0*4)));
    *((__global float4*)(data0+lidx0*4)) = (float4)(float4)(((val0).x-m),((val0).y-m),((val0).z-m),((val0).w-m));
    }}
    """).build()
    knl = prg.red
    knl(queue, (256,1), (256,1), a_g,c_g) #shape of output !!!!!
    cl.enqueue_copy(queue, c, c_g)
    return c

def minus_max(a,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c = np.zeros((12,s))
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
            data0[i + lid*{s}] = data1[i + lid*{s}] - m;
        }}
    }}
    """).build()
    knl = prg.k
    knl(queue, (12,1), (12,1), a_g,c_g) #todo hardcoded
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


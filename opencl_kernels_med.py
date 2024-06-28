import numpy as np
import pyopencl as cl
import time
import math

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
dim = 1024
n_heads = 16

def add(a,b,b_s=0,a_s=0):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    res_np = np.zeros(1024).astype(np.float32).flatten()
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
    #res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

    prg = cl.Program(ctx, f"""
    __kernel void add(
        __global const float *a, __global const float *b, __global float *res)
    {{
    int gidx0 = get_global_id(0);
        res[gidx0] = a[{a_s*1024} + gidx0] + b[gidx0 + {b_s}*1024];    
    }}
    """).build()
    knl = prg.add
    knl(queue, (1024,1), (256,1), a_g, b_g,res_g) #todo check shape
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

len = 1024
loop_size = int(len / 256)
len_short = 1024
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
    barrier(CLK_LOCAL_MEM_FENCE); //rory need this barrier when input is large like 10240, 1024 seems to work
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

    knl(queue, (1024,1), (256,1), a_g, b_g, c_g, d_g) #has to be multiple of 256
    cl.enqueue_copy(queue, a, a_g)
    return a

def matmul2(a,b,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten() #todo, shouldnt be needed
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([16,1,s])
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
    local0 = min(256,s*64*16)
    group0 = math.ceil(16*s / local0) * local0
    knl(queue, (group0,1), (local0,1), a_g, b_g,c_g)
    cl.enqueue_copy(queue, c, c_g)
    return c

def minus_max(a,s=112):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c = np.zeros((16,1,s))
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
    knl(queue, (16,1), (16,1), a_g,c_g) #todo hardcoded
    cl.enqueue_copy(queue, c, c_g)
    return c

def matmul3(a,b,s):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten() #todo, shouldnt be needed
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c = np.zeros([16,1,64])
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
    knl(queue, (64*16,1), (64,1), a_g, b_g,c_g) #todo, this is arbitrary
    cl.enqueue_copy(queue, c, c_g)
    return c

def matvec(a,b,c):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros([1024])
    d = np.float32(d)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *a, __global const float *b, __global const float *c, __global float *res)
    {{
        int lidx0 = get_global_id(0);
        float acc = 0;
        for(int x = 0; x < 1024; x++) {{
            acc += a[x] * b[x*1024 + lidx0];
        }}
        res[lidx0] = acc + c[lidx0];
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (1024,1), (256,1), a_g, b_g,c_g,d_g)
    cl.enqueue_copy(queue, d, d_g)
    return d

def matvec2(h,weight2): #pass bias in instead of adding to zero, todo for other kernels
    rows = 1024
    cols = 50257
    res = np.zeros(cols).astype(np.float32)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *h, __global const float *weight2 , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int j = 0; j < {rows}; j++) {{
            res[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
        }}
    }}
    """).build()
    knl = prg.matvec
    gidx = math.ceil(cols / 16) * 16
    knl(queue, (gidx,1), (16,1), h_g, bias2_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

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

def matmulcr(a,b): #column-row weight (b) #column-row weight (b) #todo different from main, row-column order or opposite
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
                    total += a[y*{cols} + k] * b[x + k*{rows}]; 
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
    barrier(CLK_LOCAL_MEM_FENCE); //rory need this barrier when input is large like 10240, 1024 seems to work
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

def test_equal(a,b):
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    a = a.flatten()
    b = b.flatten()
    if type(a[0]) != type(b[0]):
        exit()
    x = np.shape(a)[0]
    c = np.zeros(x).astype(np.float32)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    prg = cl.Program(ctx, f"""
    __kernel void equal(
        __global const float *a, __global const float *b, __global float *c)
    {{
        for(int i = 0; i < {x}; i++) {{
            c[i] = a[i] - b[i];
        }}
    }}
    """).build()

    knl = prg.equal
    knl(queue, (1,1), (1,1), a_g, b_g, c_g)
    cl.enqueue_copy(queue, c, c_g)
    np.testing.assert_allclose(c,np.zeros(x).astype(np.float32))
    return


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

def kernel_1b(h_in,c,d,f):
    size = np.shape(h_in)[0]
    ls = 256
    seg5 = int(size / ls)
    h_in_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_in)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    prg = cl.Program(ctx, f"""
    __kernel void knl(
        __global float *h_in, __global const float *c, __global const float *d,
        __global float *f)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg5}];
        __attribute__ ((aligned (16))) __local float mean;
        int lidx0 = get_local_id(0);
        float total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += h_in[lidx0*{seg5} + i];
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] -= mean;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += pow(h_in[lidx0*{seg5} + i],2);
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] = (h_in[i + lidx0*{seg5}] * c[i + lidx0*{seg5}]) / mean + d[i + lidx0*{seg5}];
        }}
    }}
    """).build()
    knl = prg.knl
    knl(queue, (ls,1), (ls,1), h_in_g, c_g, d_g,f_g)
    cl.enqueue_copy(queue, h_in, h_in_g)
    return h_in

def kernel_1b(h_in,c,d,f):
    size = np.shape(h_in)[0]
    ls = 256
    seg5 = int(size / ls)
    h_in_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_in)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    prg = cl.Program(ctx, f"""
    __kernel void knl(
        __global float *h_in, __global const float *c, __global const float *d,
        __global float *f)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg5}];
        __attribute__ ((aligned (16))) __local float mean;
        int lidx0 = get_local_id(0);
        float total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += h_in[lidx0*{seg5} + i];
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] -= mean;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg5}; i++) {{
            total += pow(h_in[lidx0*{seg5} + i],2);
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
        for(int i = 0; i < {seg5}; i++) {{
            h_in[i + lidx0*{seg5}] = (h_in[i + lidx0*{seg5}] * c[i + lidx0*{seg5}]) / mean + d[i + lidx0*{seg5}];
        }}
    }}
    """).build()
    knl = prg.knl
    knl(queue, (ls,1), (ls,1), h_in_g, c_g, d_g,f_g)
    cl.enqueue_copy(queue, h_in, h_in_g)
    return h_in


def matvec_b(a,b,c):
    ls = 256
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    a = a.flatten()
    b = b.flatten()
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d = np.zeros([1024])
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
    return c

def matvec2_b(h,weight2): #pass bias in instead of adding to zero, todo for other kernels
    rows = 1024
    cols = 50257
    res = np.zeros(cols).astype(np.float32)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    d = 1
    gidx = math.ceil((cols/d) / 256) * 256
    prg = cl.Program(ctx, f"""
    __kernel void matvec(
        __global const float *h, __global const float *weight2 , __global float *res)
    {{
        int gidx0 = get_global_id(0);
        for(int i = 0; i < {d}; i++) {{
            for(int j = 0; j < {rows}; j++) {{
                res[gidx0 + {math.ceil((cols/d) / 256) * 256}*i] += 
                h[j] * weight2[gidx0 + {math.ceil((cols/d) / 256) * 256}*i + j*{cols}];
            }}
        }}
    }}
    """).build()
    knl = prg.matvec
    knl(queue, (gidx,1), (256,1), h_g, bias2_g,res_g)
    cl.enqueue_copy(queue, res, res_g)
    return res

def kernel_2(h,c,d,e,f,g,keys,values,start_pos,weight,bias,\
    weight2,bias2,weight3,bias3,weight4,bias4): #g = size
    ls = 256
    zeros = np.zeros(np.shape(bias4)[0]).astype(np.float32)
    zeros2 = np.zeros(16*(start_pos+1)).astype(np.float32)
    xq = f[0:g]
    xk = f[g:2*g]
    xv = f[2*g:]
    seg = int(dim / ls) #todo
    seg3 = math.ceil(16*(start_pos+1)*(start_pos+1) / ls)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    e_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e)
    xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
    xk_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xk)
    xv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xv)
    keys_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    weight2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias2)
    weight3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight3)
    bias3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias3)
    weight4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight4)
    bias4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias4)
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    h_temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros2)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d, __global const float *e,
        __global float *xq, __global const float *xk, __global const float *xv, __global float *keys,
        __global float *values,
        __global const float *weight,__global const float *bias,
        __global const float *weight2, __global const float *bias2,
        __global const float *weight3, __global float *bias3,
        __global const float *weight4,
        __global float *bias4, __global float *h_temp, __global float *h,
        __global float *temp3)
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
            mean = total / {dim};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] -= mean;
            total += pow(a[lidx0*{seg} + i],2);
        }}
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = pow(total / {dim} + 1e-5,0.5);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {int(dim*3 / ls)}; i++) {{
            float total = 0;
            for(int k = 0; k < {dim}; k++) {{
                total += ((a[k] * c[k]) / mean + d[k]) * e[(lidx0*{int(dim*3 / ls)} + i)*{dim} + k];
            }}
            if((lidx0*{int(dim*3 / ls)} + i) < {g}) {{
                xq[lidx0*{int(dim*3 / ls)} + i] += total;
                }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {g} && (lidx0*{int(dim*3 / ls)} + i) < {2*g}) {{
                keys[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {g}] = xk[lidx0*{int(dim*3 / ls)} + i - {g}] + total;
            }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {2*g}) {{
                values[{start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {2*g}] = xv[lidx0*{int(dim*3 / ls)} + i - {2*g}] + total;
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int z = 0; z < {seg3}; z++) {{
            int x = (z + lidx0*{seg3}) % {start_pos+1};
            int k = (z + lidx0*{seg3}) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xq[i + 64*k] * keys[x*16*64 + i + 64*k];
            }}                  
            temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0 < 16){{
        float m = -INFINITY;
        for(int i = 0; i < {start_pos+1}; i++) {{
            float val = temp3[i + lidx0*{start_pos+1}];
            m = max(m,val);
        }}
        float t = 0;
        for(int i = 0; i < {start_pos+1}; i++) {{
            temp3[i + lidx0*{start_pos+1}] = exp(temp3[i + lidx0*{start_pos+1}] - m);
            float val = temp3[i + lidx0*{start_pos+1}];
            t = t+val;
        }}
        for(int i = 0; i < {start_pos+1}; i++) {{
            temp3[i + lidx0*{start_pos+1}] = temp3[i + lidx0*{start_pos+1}] / t;
        }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int g = 0; g < {seg}; g++) {{
            int y = (g + lidx0*{seg}) / 64;
            int x = (g + lidx0*{seg}) % 64;
            float acc0 = 0;
            for(int i = 0; i < {start_pos+1}; i++) {{
                acc0 += temp3[i + {start_pos+1}*y] * values[i*16*64 + x + y*64];
            }}
            xq[x + y*64] = acc0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            float acc = 0;
            for(int x = 0; x < {dim}; x++) {{
                acc += xq[x] * weight[x*{dim} + lidx0*{seg} + i];
            }}
            h[lidx0*{seg} + i] = a[lidx0*{seg} + i] + acc + bias[lidx0*{seg} + i];
            h_temp[lidx0*{seg} + i] = h[lidx0*{seg} + i];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            total += h[lidx0*{seg} + i];
        }}
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = total / {dim};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            h[i + lidx0*{seg}] = h[i + lidx0*{seg}] - mean;
            total += pow(h[lidx0*{seg} + i],2);
        }}        
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = pow(total / {dim} + 1e-5,0.5);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {int(dim*4 / ls)}; i++) {{
            for(int j = 0; j < {dim}; j++) {{
                bias3[i + lidx0*{int(dim*4 / ls)}] += ((h[j] * weight2[j]) / mean + bias2[j]) * weight3[(i + lidx0*{int(dim*4 / ls)})*{dim} + j];
            }}
            bias3[i + lidx0*{int(dim*4 / ls)}] = 0.5 * bias3[i + lidx0*{int(dim*4 / ls)}]\
            * (1 + tanh(bias3[i + lidx0*{int(dim*4 / ls)}] * 0.7978845608\
            * (1 + 0.044715 * pow(bias3[i + lidx0*{int(dim*4 / ls)}],2))));
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  
        for(int i = 0; i < {int(np.shape(bias4)[0] / ls)}; i++) {{
            for(int j = 0; j < {dim*4}; j++) {{
                bias4[lidx0 + i*{ls}] += bias3[j] * weight4[lidx0 + i*{ls} + j*{dim}];
            }}
            a[lidx0 + i*{ls}] = bias4[lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1),a_g,c_g,d_g,e_g,xq_g,xk_g,xv_g\
    ,keys_g,values_g,weight_g,bias_g,\
    weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,h_g,h_temp_g,temp_g)
    cl.enqueue_copy(queue, keys, keys_g)
    cl.enqueue_copy(queue, values, values_g)
    cl.enqueue_copy(queue, h, a_g)
    return h

def minus_max_b(a):
    s = np.shape(a)[2] #todo remove dim
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    ls = 256

    prg = cl.Program(ctx, f"""
    __kernel void k(
        __global float *data0)
    {{
        int lidx0 = get_local_id(0);
        if(lidx0 < 16){{
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
        if(lidx0 < 16) {{
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

def matmul3_b(xq,values,s):
    xq = xq.flatten()
    xq_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xq)
    values = values.flatten() #todo, shouldnt be needed
    values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
    res = np.zeros([16,1,64]).astype(np.float32)
    res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)
    ls = 256
    seg = int((16*64) / ls)
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
            for(int j = 0; j < 16; j++) {{
                for(int k = 0; k < 64; k++) {{
                    //at[j*64*{s} + i*64 + k] = a[i*16*64 + j*64 + k];
                    at[j*64*{s} + i + k*{s}] = a[i*16*64 + j*64 + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (1,1), (1,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at.reshape(16,64,s)

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
            int y = (lidx0*{seg2} + i) / 16;
            int x = (lidx0*{seg2} + i) % 16;
            for(int k = 0; k < 64; k++) {{
                if((y*16*64 + x*64 + k) < {np.shape(a)[0]}) {{
                    //at[x*64*{s} + y*64 + k] = a[y*16*64 + x*64 + k];
                    at[x*64*{s} + y + k*{s}] = a[y*16*64 + x*64 + k];
                }}
            }}
        }}
    }}
    """).build()
    knl = prg.matmul
    knl(queue, (256,1), (256,1), a_g, at_g)
    cl.enqueue_copy(queue, at, at_g)
    return at.reshape(16,64,s)

def kernel_3(h,weight,bias):
    size = np.shape(h)[0]
    ls = 256
    seg = int(size / ls) #todo
    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *h, __global const float *weight, __global const float *bias)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg}];
        __attribute__ ((aligned (16))) __local float mean;
        int lidx0 = get_local_id(0);
        float total = 0;
        for(int i = 0; i < {seg}; i++) {{
            total += h[lidx0*{seg} + i];
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
            h[i + lidx0*{seg}] -= mean;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            total += pow(h[lidx0*{seg} + i],2);
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
            h[i + lidx0*{seg}] = (h[i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / mean + bias[i + lidx0*{seg}];
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1), h_g, weight_g, bias_g) #rory to test large stuff
    cl.enqueue_copy(queue, h, h_g)
    return h

def kernel_4(h,c,d,f,g,start_pos,bias,\
    weight2,bias2,bias3,\
    e,keys_values,weight,weight3,weight4,bias4): #g = size
    ls = 256
    zeros = np.zeros(np.shape(bias4)[0]).astype(np.float32)
    zeros2 = np.zeros(16*(start_pos+1)).astype(np.float32)
    seg = int(dim / ls) #todo
    seg3 = math.ceil(16*(start_pos+1)*(start_pos+1) / ls)
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h)
    c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    e_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e)
    keys_values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys_values) #AND VALUES NOW
    weight_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    bias_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    weight2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight2)
    bias2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias2)
    weight3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight3)
    bias3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias3)
    weight4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight4)
    bias4_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias4)

    h_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    h_temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
    temp_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros2)
    xqkv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
    prg = cl.Program(ctx, f"""
    __kernel void mm(
        __global float *a, __global const float *c, __global const float *d, __global const float *e,
        __global float *xqkv, __global float *keys_values,
        __global const float *weight,__global const float *bias,
        __global const float *weight2, __global const float *bias2,
        __global const float *weight3, __global float *bias3,
        __global const float *weight4,
        __global float *bias4, __global float *h_temp, __global float *h,
        __global float *temp3)
    {{
        __attribute__ ((aligned (16))) __local float temp[{seg}];
        __attribute__ ((aligned (16))) __local float mean;
        int lidx0 = get_local_id(0);
        for(int r = 0; r < 4; r++) {{
        barrier(CLK_LOCAL_MEM_FENCE);  
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
            mean = total / {dim};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            a[i + lidx0*{seg}] -= mean;
            total += pow(a[lidx0*{seg} + i],2);
        }}
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = pow(total / {dim} + 1e-5,0.5);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {int(dim*3 / ls)}; i++) {{
            float total = 0;
            for(int k = 0; k < {dim}; k++) {{
                total += ((a[k] * c[1024*r + k]) / mean + d[1024*r + k]) * e[1024*1024*3*r + (lidx0*{int(dim*3 / ls)} + i)*{dim} + k];
            }}
            if((lidx0*{int(dim*3 / ls)} + i) < {g}) {{
                xqkv[1024*3*r + lidx0*{int(dim*3 / ls)} + i] += total;
                }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {g} && (lidx0*{int(dim*3 / ls)} + i) < {2*g}) {{
                keys_values[128*16*64*2*r + {start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {g}] = xqkv[1024*3*r+1024 + lidx0*{int(dim*3 / ls)} + i - {g}] + total;
            }}
            if((lidx0*{int(dim*3 / ls)} + i) >= {2*g}) {{
                keys_values[128*16*64*2*r + 128*16*64 + {start_pos}*{dim} + lidx0*{int(dim*3 / ls)} + i - {2*g}] = xqkv[1024*3*r+1024*2 + lidx0*{int(dim*3 / ls)} + i - {2*g}] + total;
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int z = 0; z < {seg3}; z++) {{
            int x = (z + lidx0*{seg3}) % {start_pos+1};
            int k = (z + lidx0*{seg3}) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xqkv[1024*3*r + i + 64*k] * keys_values[128*16*64*2*r + x*16*64 + i + 64*k];
            }}                  
            temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(self.head_dim)
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0 < 16){{
        float m = -INFINITY;
        for(int i = 0; i < {start_pos+1}; i++) {{
            float val = temp3[i + lidx0*{start_pos+1}];
            m = max(m,val);
        }}
        float t = 0;
        for(int i = 0; i < {start_pos+1}; i++) {{
            temp3[i + lidx0*{start_pos+1}] = exp(temp3[i + lidx0*{start_pos+1}] - m);
            float val = temp3[i + lidx0*{start_pos+1}];
            t = t+val;
        }}
        for(int i = 0; i < {start_pos+1}; i++) {{
            temp3[i + lidx0*{start_pos+1}] = temp3[i + lidx0*{start_pos+1}] / t;
        }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int g = 0; g < {seg}; g++) {{
            int y = (g + lidx0*{seg}) / 64;
            int x = (g + lidx0*{seg}) % 64;
            float acc0 = 0;
            for(int i = 0; i < {start_pos+1}; i++) {{
                acc0 += temp3[i + {start_pos+1}*y] * keys_values[128*16*64*2*r + 128*16*64 + i*16*64 + x + y*64];
            }}
            xqkv[1024*3*r + x + y*64] = acc0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {seg}; i++) {{
            float acc = 0;
            for(int x = 0; x < {dim}; x++) {{
                acc += xqkv[1024*3*r + x] * weight[1024*1024*r + x*{dim} + lidx0*{seg} + i];
            }}
            h[lidx0*{seg} + i] = a[lidx0*{seg} + i] + acc + bias[1024*r + lidx0*{seg} + i];
            h_temp[lidx0*{seg} + i] = h[lidx0*{seg} + i];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            total += h[lidx0*{seg} + i];
        }}
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = total / {dim};  
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        total = 0;
        for(int i = 0; i < {seg}; i++) {{
            h[i + lidx0*{seg}] = h[i + lidx0*{seg}] - mean;
            total += pow(h[lidx0*{seg} + i],2);
        }}        
        temp[lidx0] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lidx0==0) {{
            total = 0;
            for(int i = 0; i < {ls}; i++) {{
                total += temp[i];
            }}
            mean = pow(total / {dim} + 1e-5,0.5);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < {int(dim*4 / ls)}; i++) {{
            for(int j = 0; j < {dim}; j++) {{
                bias3[1024*4*r + i + lidx0*{int(dim*4 / ls)}] += ((h[j] * weight2[1024*r + j]) / mean + bias2[1024*r + j]) * weight3[1024*1024*4*r + (i + lidx0*{int(dim*4 / ls)})*{dim} + j];
            }}
            bias3[1024*4*r + i + lidx0*{int(dim*4 / ls)}] = 0.5 * bias3[1024*4*r + i + lidx0*{int(dim*4 / ls)}]\
            * (1 + tanh(bias3[1024*4*r + i + lidx0*{int(dim*4 / ls)}] * 0.7978845608\
            * (1 + 0.044715 * pow(bias3[1024*4*r + i + lidx0*{int(dim*4 / ls)}],2))));
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  
        for(int i = 0; i < {math.ceil(np.shape(bias4)[0] / 4 / ls)}; i++) {{ //todo because there's 2 now
            for(int j = 0; j < {dim*4}; j++) {{
                bias4[1024*r + lidx0 + i*{ls}] += bias3[1024*4*r + j] * weight4[1024*4*1024*r + lidx0 + i*{ls} + j*{dim}];
            }}
            a[lidx0 + i*{ls}] = bias4[1024*r + lidx0 + i*{ls}] + h_temp[lidx0 + i*{ls}];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);  
        }}
    }}
    """).build()
    knl = prg.mm
    knl(queue, (ls,1), (ls,1),a_g,c_g,d_g,e_g,xqkv_g\
    ,keys_values_g,weight_g,bias_g,\
    weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,h_g,h_temp_g,temp_g)
    cl.enqueue_copy(queue, keys_values, keys_values_g)
    cl.enqueue_copy(queue, h, a_g)
    return h
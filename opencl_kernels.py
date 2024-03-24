import numpy as np
import pyopencl as cl

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def add(a,b):
  dim = len(a)
  a = a.flatten()
  b = b.flatten()
  a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
  b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

  res_np = np.empty(dim).astype(np.float32).flatten()
  res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * 4))

  prg = cl.Program(ctx, f"""
  __kernel void add(
      __global const float *a, __global float *res)
  {{
  }}
  """).build()
  knl = prg.add
  knl(queue, (1,1), (1,1), a_g, res_g) #todo check shape
  cl.enqueue_copy(queue, res_np, res_g)
  return res_np

a = np.zeros(1024)
b = np.zeros(1024)
a.fill(4)
b.fill(6)
c = add(a,b)
print(c)
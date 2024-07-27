import pyopencl as cl
import math
import numpy as np
try:
   import Metal
except ImportError:
   pass


class buffer:
    def __init__(self,data,size,d):
        self.data = data
        self.size = size
        self.d = d
        #TODO cache np if faster?

    def np(self,params=None):
        if self.d == "Metal":
            output = np.asarray(self.data.contents().as_buffer(self.size))
            return np.frombuffer(output, dtype=np.float32)
        if self.d == "OpenCL":
            queue = params["queue"]
            ret = np.zeros(math.ceil(self.size/4)).astype(np.float32)
            cl.enqueue_copy(queue, ret, self.data)
            return ret

def create_buffer(a,d,params):
  if d == "Metal":
    device = params["device"]
    a_buffer = device.newBufferWithLength_options_(len(a.flatten())*4 ,1)
    m = a_buffer.contents().as_buffer(len(a.flatten())*4)
    m[:] = bytes(a)
    return buffer(a_buffer,len(a.flatten())*4,d)
  if d == "OpenCL":
    ctx = params["ctx"]
    mf = params["mf"]
    data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    return buffer(data,len(a.flatten()),d)
  return None
  
def create_buffer_empty(size,d,params):
  if d == "Metal":
    device = params["device"]
    a_buffer = device.newBufferWithLength_options_(size ,1)
    return buffer(a_buffer,size,d)
  if d == "OpenCL":
    ctx = params["ctx"]
    mf = params["mf"]
    data = cl.Buffer(ctx, mf.READ_ONLY, size)
    return buffer(data,size,d)
  return None
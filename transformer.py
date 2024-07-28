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

def compile(prg_str,d,params):
  if d == "Metal":
    library, err = params["device"].newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
    return library
  if d == "OpenCL":
    return cl.Program(params["ctx"],prg_str).build()
  
def run(prg,func,params,args,gs,ls,d):
  if d == "Metal":
    mtl_queue = params["queue"]
    device = params["device"]
    fxn = prg.newFunctionWithName_(func)
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    encoder.setComputePipelineState_(pipeline_state)
    i = 0
    for arg in args:
        encoder.setBuffer_offset_atIndex_(arg.data, 0, i)
        i+=1
    threadsPerGrid = Metal.MTLSizeMake(gs,1,1)
    threadsPerThreadGroup = Metal.MTLSizeMake(ls,1,1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup)
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
  if d == "OpenCL":
     gs*=ls
     queue = params["queue"]
     kernel = getattr(prg,func)
     kernel(queue, (gs,1), (ls,1),*args)
  return

def run_old(prg,func,params,args,gs,ls,d):
  if d == "Metal":
    mtl_queue = params["queue"]
    device = params["device"]
    fxn = prg.newFunctionWithName_(func)
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    encoder.setComputePipelineState_(pipeline_state)
    i = 0
    for arg in args:
        encoder.setBuffer_offset_atIndex_(arg.data, 0, i)
        i+=1
    threadsPerGrid = Metal.MTLSizeMake(gs,1,1)
    threadsPerThreadGroup = Metal.MTLSizeMake(ls,1,1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup)
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
  if d == "OpenCL":
     queue = params["queue"]
     kernel = getattr(prg,func)
     kernel(queue, (gs,1), (ls,1),*args)
  return

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
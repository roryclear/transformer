import pyopencl as cl
import math
import numpy as np
try:
   import Metal
except ImportError:
   pass

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except ImportError:
    pass


class buffer:
    def __init__(self,data,size,d):
        self.data = data
        self.size = size
        self.d = d
        self.np_data = None #cache?
        #TODO cache np if faster?

    def np(self,params=None):
        if self.d == "Metal":
            output = np.asarray(self.data.contents().as_buffer(self.size))
            return np.frombuffer(output, dtype=np.float32)
        if self.d == "OpenCL":
            queue = params["queue"]
            if self.np_data is None:
              self.np_data = np.zeros(math.ceil(self.size/4)).astype(np.float32)
            cl.enqueue_copy(queue, self.np_data, self.data)
            return self.np_data
        if self.d == "CUDA":
          if self.np_data is None:
              self.np_data = np.zeros(math.ceil(self.size/4)).astype(np.float32)
          cuda.memcpy_dtoh(self.np_data, self.data)
          return self.np_data
        
    def copy(self,params):
      return create_buffer(self.np(params),self.d,params)
        
    def rand_like(x,params):
          return create_buffer(np.random.random(np.shape(x.np(params).flatten())).astype(np.float32),x.d,params)
    
    def delete(self): #todo OpenCL
       if self.d == "Metal":
          self.data.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
          self.data.release()

          
def compile(prg_str,d,params):
  if d == "Metal":
    library, err = params["device"].newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
    return library
  if d == "OpenCL":
    return cl.Program(params["ctx"],prg_str).build()
  if d == "CUDA":
    return SourceModule(prg_str)
  
def run(prg,func,params,args,gs,ls,d):
  if d == "Metal":
    gs = math.ceil(gs/ls)
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
     data = []
     for a in args: data.append(a.data) #todo, better way?
     kernel(queue, (gs,1), (ls,1),*data)
  if d == "CUDA":
    gs = math.ceil(gs/ls)
    fxn = prg.get_function(func)
    data = []
    for a in args: data.append(a.data) #todo, better way?
    fxn(*data,block=(ls,1,1),grid=(gs,1))
  return

def run_test(prg,func,params,args,gs,ls,d): #TODO, only for metal because no delete in OpenCL yet
  args_copy_a = []
  for a in args:
    args_copy_a.append(a.copy(params))
  run(prg,func,params,args_copy_a,gs,ls,d) 
  print("test =",func)
  for j in range(len(args_copy_a)):
    assert(np.isnan(np.max(args_copy_a[j].np(params))) == False)
    assert(np.isinf(np.max(args_copy_a[j].np(params))) == False)
  for x in range(3):
    print("test =",x,func)
    args_copy_b = []
    for a in args: 
        args_copy_b.append(a.copy(params))
    run(prg,func,params,args_copy_b,gs,ls,d)
    for j in range(len(args_copy_b)):
      np.testing.assert_allclose(args_copy_a[j].np(params),args_copy_b[j].np(params),1e-6)
    for j in range(len(args_copy_b)):
      args_copy_b[j].delete()
    args_copy_b = [] #todo, needed?
  for j in range(len(args_copy_a)):
    args_copy_a[j].delete()
  args_copy_a = []#todo, needed?
  run(prg,func,params,args,gs,ls,d)
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
    a_buffer = device.newBufferWithLength_options_(a.nbytes ,1)
    m = a_buffer.contents().as_buffer(a.nbytes)
    m[:] = bytes(a)
    return buffer(a_buffer,a.nbytes,d)
  if d == "OpenCL":
    ctx = params["ctx"]
    mf = params["mf"]
    data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    return buffer(data,a.nbytes,d)
  if d == "CUDA":
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    return buffer(a_gpu,a.nbytes,d)
  
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
  if d == "CUDA":
    a = np.zeros(math.ceil(size/4)).astype(np.float32)
    a_gpu = cuda.mem_alloc(size) #TODO, this won't be all zeros, so need to copy zeros in for now
    cuda.memcpy_htod(a_gpu, a)
    return buffer(a_gpu,size,d)
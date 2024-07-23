import numpy as np
import time
import math
import Metal

test = False

ls = 256
ls3 = 256 #threadExecutionWidth correct value to use

class buffer:
    def __init__(self,data,size):
        self.data = data
        self.size = size

    def np(self):
        output = np.asarray(self.data.contents().as_buffer(self.size))
        return np.frombuffer(output, dtype=np.float32)

def create_metal_buffer(a,device):
  a_buffer = device.newBufferWithLength_options_(len(a.flatten())*4 ,1)
  m = a_buffer.contents().as_buffer(len(a.flatten())*4)
  m[:] = bytes(a)
  return buffer(a_buffer,len(a.flatten())*4)

def create_metal_buffer_empty(size,device):
  a_buffer = device.newBufferWithLength_options_(size ,1)
  return buffer(a_buffer,size)

class Metal_Kernels:
    def __init__(self,dim,n_heads,max_context):
        self.prg_cache = {}
        self.dim = dim
        self.n_heads = n_heads
        self.max_context = max_context
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.mtl_queue = self.device.newCommandQueue()

    def run_metal(self,fxn,gs,ls,args,device=None):
        command_buffer = self.mtl_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        pipeline_state, err = self.device.newComputePipelineStateWithFunction_error_(fxn, None)
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
        return
    

    def run_metal2(self,fxn,g,ls,args,device=None):
        command_buffer = self.mtl_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        pipeline_state, err = self.device.newComputePipelineStateWithFunction_error_(fxn, None)
        encoder.setComputePipelineState_(pipeline_state)
        i = 0
        for arg in args:
            encoder.setBuffer_offset_atIndex_(arg.data, 0, i)
            i+=1
        threadsPerGrid = Metal.MTLSizeMake(g(ls),1,1)
        threadsPerThreadGroup = Metal.MTLSizeMake(ls,1,1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        return
    

    def run_metal_test(self,fxn,gs,ls,args,device=None):
        args_copy = []
        n = 5
        f = 0
        for i in range(len(args)): args_copy.append(args[i].np())
        excepted_output = []
        for x in range(n):
            ff = False
            args_buffers = []
            for j in range(len(args_copy)): args_buffers.append(create_metal_buffer(args_copy[j],self.device))
            command_buffer = self.mtl_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            pipeline_state, err = self.device.newComputePipelineStateWithFunction_error_(fxn, None)
            encoder.setComputePipelineState_(pipeline_state)
            i = 0
            for arg in args_buffers:
                encoder.setBuffer_offset_atIndex_(arg.data, 0, i)
                i+=1
            threadsPerGrid = Metal.MTLSizeMake(gs,1,1)
            threadsPerThreadGroup = Metal.MTLSizeMake(ls,1,1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup)
            encoder.endEncoding()
            t = time.perf_counter()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            t = time.perf_counter() - t
            if len(excepted_output) == 0:
                for j in range(len(args_buffers)):
                    excepted_output.append(np.copy(args_buffers[j].np()))
            else:
                for j in range(len(excepted_output)):
                    np.testing.assert_allclose(excepted_output[j],args_buffers[j].np(),rtol=1e-6)
                        #ff = True
                    excepted_output[j] = np.copy(args_buffers[j].np())
                if ff: f+=1
                    #np.testing.assert_allclose(excepted_output[j],args_buffers[j].np(),rtol=1e-6)
        print("flakiness =",f/n)
        args_copy = None
        excepted_output = None
        args_buffers = None
        self.run_metal(fxn,gs,ls,args,device)
        return
    
    def run_metal_test2(self,fxn,g,ls,args,device=None):
        ls = 1
        while ls < 2048:
            print("ls =",ls)
            self.run_metal_test(fxn,g(ls),ls,args,device)
            ls*=2
        return None

    def add(self,a_g,b_g,b_s=0,a_s=0):
        if hasattr(self, 'add_res_g') == False:
            self.add_res_g = create_metal_buffer_empty(self.dim*4,self.device)
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void add(
            device const float *a, device const float *b, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
        int gidx0 = gid.x;
        if(gidx0 < {self.dim}) {{
            res[gidx0] = a[{int(a_s)*self.dim} + gidx0] + b[gidx0 + {b_s*self.dim}];
        }}   
        }}
        """
        prg, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
        fxn = prg.newFunctionWithName_("add")
        def g(ls): return math.ceil(self.dim / ls)
        self.run_metal2(fxn,g,ls,[a_g, b_g,self.add_res_g],self.device)
        return self.add_res_g

    def tok_emb(self,tokens,weight_g,weight_2_g,no_tokens):
        tokens_g = create_metal_buffer(tokens.astype(np.int32),self.device)
        size = no_tokens*self.dim
        tok_emb_g = create_metal_buffer_empty(no_tokens*self.dim*4,self.device)
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm(
            device int *tokens, device const float *weight, device const float *weight2,  device float *tok_emb, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = gidx0 / {self.dim};
            int j = gidx0 % {self.dim};
            tok_emb[i*{self.dim} + j] = weight[tokens[i]*{self.dim} + j] + weight2[i*{self.dim} + j];
        }}
        """

        options = Metal.MTLCompileOptions.alloc().init()
        library, err = self.device.newLibraryWithSource_options_error_(prg_str, options, None)
        fxn = library.newFunctionWithName_("mm")
        def gs(ls): return math.ceil(size / ls)
        self.run_metal2(fxn,gs,ls3,[tokens_g,weight_g,weight_2_g,tok_emb_g])
        return tok_emb_g

    def kernel_1(self,h_g,weight_g,bias_g,weight2_g,temperature,random_num):
        ls = 256
        seg = int(self.dim / ls3)
        rows = self.dim
        cols = 50257
        if hasattr(self, 'logits_g') == False:
            self.logits_g = create_metal_buffer_empty(50257*4,self.device)
        if hasattr(self, 'res') == False:
            self.res = np.zeros(1).astype(np.float32)
        if hasattr(self, 'res_g') == False:
            self.res_g = create_metal_buffer_empty(1*4,self.device)
        seg2 = math.ceil(50257 / ls)
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm4(
            device float *h, device const float *weight, device const float *bias, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls3}];
            threadgroup float mean;
            int lidx0 = gid.x;
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += h[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls3}; i++) {{
                    total += temp[i];
                }}
                mean = total / {self.dim};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] -= mean;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(h[lidx0*{seg} + i],2);
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls3}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {self.dim} + 1e-5,0.5);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                h[i + lidx0*{seg}] = (h[i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / mean + bias[i + lidx0*{seg}];
            }}
        }}
        kernel void matvec(
            device const float *h, device const float *weight2 , device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            res[gidx0] = 0;
            for(int j = 0; j < {rows}; j++) {{
                res[gidx0] += h[j] * weight2[gidx0 + j*{cols}];
            }}
            res[gidx0] /= {temperature};
        }}
        kernel void mm5(
            device const float *a, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        kernel void mm6(
        device float *a, device float *res, uint3 gid [[thread_position_in_grid]])
        {{  
            int gidx0 = gid.x;
            a[gidx0] = exp(a[gidx0] - res[0]);
        }}

        kernel void mm8(
        device float *a, device const float *res, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            a[gidx0] = a[gidx0] / res[0];
        }}

        kernel void mm9(
        device const float *a, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls3}];
            int lidx0 = gid.x;
            temp[lidx0] = 0;
            for(int i = 0; i < {math.ceil(50257 / ls3)}; i++) {{
                if(lidx0*{math.ceil(50257 / ls3)} + i < 50257){{
                temp[lidx0] += a[lidx0*{math.ceil(50257 / ls3)} + i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float t = 0;
            if(lidx0 == 0) {{
                for(int i = 0; i < {ls3}; i++) {{
                    t+=temp[i];
                }}
                res[0] = t;
            }}
        }}

        kernel void mm10(
        device float *a, uint3 gid [[thread_position_in_grid]])
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}
        """
        
        if prg_str not in self.prg_cache:
            library, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]

        prg_str = f"""
        kernel void mm11(
        device float *a, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            if((a[gidx0] / a[50256]) < {random_num}) {{
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}
        """
        prg2, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)

        fxn = prg.newFunctionWithName_("mm4")
        self.run_metal(fxn,1,ls3,[h_g, weight_g, bias_g])

        fxn = prg.newFunctionWithName_("matvec")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[h_g, weight2_g,self.logits_g])

        fxn = prg.newFunctionWithName_("mm5")
        self.run_metal(fxn,1,1,[self.logits_g,self.res_g])

        fxn = prg.newFunctionWithName_("mm6")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[self.logits_g,self.res_g])

        fxn = prg.newFunctionWithName_("mm5")
        self.run_metal(fxn,1,1,[self.logits_g,self.res_g])

        fxn = prg.newFunctionWithName_("mm8")
        self.run_metal2(fxn,gs,ls3,[self.logits_g,self.res_g])

        fxn = prg.newFunctionWithName_("mm9")
        self.run_metal(fxn,1,ls3,[self.logits_g,self.res_g])

        fxn = prg.newFunctionWithName_("mm8")
        self.run_metal2(fxn,gs,ls3,[self.logits_g,self.res_g]) 

        fxn = prg.newFunctionWithName_("mm10")
        self.run_metal(fxn,1,1,[self.logits_g])
        
        fxn = prg2.newFunctionWithName_("mm11")
        self.run_metal2(fxn,gs,ls3,[self.logits_g])

        fxn = prg.newFunctionWithName_("mm9")        
        self.run_metal(fxn,1,ls3,[self.logits_g,self.res_g])

        res = np.asarray(self.res_g.data.contents().as_buffer(self.res_g.size))
        res = np.frombuffer(res, dtype=np.float32)
        return res

    def kernel_3(self,x_g,weight_g,bias_g,attn_weight_g,attn_bias_g,new_cache_g\
        ,ln_f_weight_g,ln_f_bias_g,n_tokens,max_content,lm_head_weight_g,temperature,random_num):
        ls = 256
        size = self.dim 
        b_cols2 = 50257
        b_rows2 = self.dim
        seg2 = math.ceil(50257 / ls)
        b_cols = self.dim*3 #todo
        b_rows = self.dim
        seg = int(size / ls) #todo
        x0_g = create_metal_buffer_empty(n_tokens*self.dim*4,self.device)
        logits_g = create_metal_buffer_empty(50257*4,self.device)
        c_g = create_metal_buffer_empty(n_tokens*b_cols*4,self.device)
        res = np.zeros(1).astype(np.float32)
        res_g = create_metal_buffer_empty(1*4,self.device)
        
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm(device const float *x_in,
            device float *x, device const float *weight, device const float *bias, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            threadgroup float temp2[{n_tokens}];
            int gidx0 = gid.x;
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; 
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + lidx0*{seg} + i] = x_in[{self.dim}*r + lidx0*{seg} + i];
                temp2[r] += x[{self.dim}*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = temp2[r];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {size};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] -= temp2[r];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[{self.dim}*r + lidx0*{seg} + i],2);
            }}
            temp[lidx0] = temp2[r];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{n_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {size} + 1e-5,0.5);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
            }}
        }}
        kernel void mm2(
            device const float *x, device const float *attn_weight, device const float *attn_bias,device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < {b_cols*n_tokens}) {{ //TODO
            int gidx0 = gid.x;
            int i = gidx0 / {n_tokens};
            int y = gidx0 % {n_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += x[y*{b_rows} + k] * attn_weight[i*{b_rows} + k]; 
            }}
            res[y*{b_cols} + i] = total + attn_bias[i];
            }}
        }}
        kernel void mm3(
            device const float *xqkv, device float *new_cache, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim*1} + j];
            new_cache[{max_content*self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*2 + j]; 
        }}
         kernel void mm4(
            device const float *xqkv, device float *new_cache, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}];
            new_cache[{max_content*self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + j + {self.dim}*2]; 
        }}
        kernel void mm5(
            device float *x, device const float *ln_f_weight, device const float *ln_f_bias, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            threadgroup float mean;
            int lidx0 = gid.x;
            float total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += x[lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = total / {size};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[i + lidx0*{seg} + {(n_tokens - 1)*self.dim}] -= mean;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(x[lidx0*{seg} + i + {(n_tokens - 1)*self.dim}],2);
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                mean = pow(total / {size} + 1e-5,0.5);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[i + lidx0*{seg}] = (x[i + lidx0*{seg} + {(n_tokens - 1)*self.dim}] * ln_f_weight[i + lidx0*{seg}]) / mean + ln_f_bias[i + lidx0*{seg}];
            }}
        }}
        kernel void matmul(
            device const float *a, device const float *b, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < {b_cols2}) {{
                int x = gid.x;
                float total = 0;
                for(int k = 0; k < {b_rows2}; k++) {{
                    total += a[k] * b[x*{b_rows2} + k]; 
                }}
                res[x] = total / {temperature}; 
            }}
        }}
        kernel void mm6(
            device const float *a, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            res[0] = a[0]; //todo why is this needed?, used to be a MAX
        }}

        kernel void mm7(
        device float *a, device const float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < 50257) {{ //TODO
            int gidx0 = gid.x;
            a[gidx0] = exp(a[gidx0] - res[0]);
            }}
        }}

        kernel void mm9(
        device float *a, device const float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < 50257) {{
            int gidx0 = gid.x;
            a[gidx0] = a[gidx0] / res[0];
            }}
        }}

        kernel void mm10(
        device const float *a, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            int lidx0 = gid.x;
            float t = 0;
            for(int i = 0; i < {seg2}; i++) {{
                if(lidx0*{seg2} + i < 50257) {{
                t += a[lidx0*{seg2} + i];
                }}
            }}
            temp[lidx0] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0 == 0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                res[0] = t;
            }}
        }}

        kernel void mm11(
        device float *a, uint3 gid [[thread_position_in_grid]])
        {{
            for(int i = 1; i < 50257; i++) {{
                a[i] += a[i-1];
            }}
        }}

        kernel void mm12(
        device float *a, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            if((a[gidx0] / a[50256]) < {random_num}) {{
                a[gidx0] = 1;
            }} else {{
                a[gidx0] = 0;
            }}
        }}
        """
        
        if prg_str not in self.prg_cache:
            library, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]


        fxn = prg.newFunctionWithName_("mm")
        def gs(ls=None): return n_tokens
        self.run_metal2(fxn,gs,ls,[x_g, x0_g, weight_g, bias_g]) #TODO

        fxn = prg.newFunctionWithName_("mm2")
        def gs(ls): return math.ceil(b_cols*n_tokens / ls)
        self.run_metal2(fxn,gs,ls3,[x0_g, attn_weight_g,attn_bias_g,c_g])

        fxn = prg.newFunctionWithName_("mm4")
        def gs(ls): return math.ceil((n_tokens*self.n_heads*64) / ls)
        self.run_metal2(fxn,gs,ls3,[c_g, new_cache_g])
        
        fxn = prg.newFunctionWithName_("mm5")
        self.run_metal(fxn,1,ls,[x_g, ln_f_weight_g, ln_f_bias_g]) #TODO, reduce
        
        fxn = prg.newFunctionWithName_("matmul")
        def gs(ls): return math.ceil(b_cols2 / ls)
        self.run_metal2(fxn,gs,ls3,[x_g, lm_head_weight_g,logits_g])

        fxn = prg.newFunctionWithName_("mm6")
        def gs(ls=None): return 1
        self.run_metal2(fxn,gs,1,[logits_g,res_g])

        fxn = prg.newFunctionWithName_("mm7")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[logits_g,res_g])

        fxn = prg.newFunctionWithName_("mm6")
        def gs(ls=None): return 1
        self.run_metal2(fxn,gs,1,[logits_g,res_g])

        fxn = prg.newFunctionWithName_("mm9")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[logits_g,res_g])

        fxn = library.newFunctionWithName_("mm10")
        def gs(ls=None): return 1
        self.run_metal2(fxn,gs,ls,[logits_g,res_g]) #TODO, need to calculate seg based on ls

        fxn = library.newFunctionWithName_("mm9")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[logits_g,res_g]) #TODO
        
        fxn = library.newFunctionWithName_("mm11")
        self.run_metal(fxn,1,1,[logits_g])

        fxn = library.newFunctionWithName_("mm12")
        def gs(ls): return math.ceil(50257 / ls)
        self.run_metal2(fxn,gs,ls3,[logits_g]) #TODO
    
        fxn = library.newFunctionWithName_("mm10")
        def gs(ls=None): return 1
        self.run_metal2(fxn,gs,ls,[logits_g,res_g]) #TODO, need to calculate seg based on ls

        res = np.asarray(res_g.data.contents().as_buffer(res_g.size))
        res = np.frombuffer(res, dtype=np.float32)
        return res

    def kernel_0(self,a_g,c_g,d_g,e_g,xqkv_g,g,keys_values_g,start_pos,weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,j=0):
        ls = 256
        ls3 = 256 #TODO why is 256 fastet than 32?
        seg3 = math.ceil(self.dim / ls3) #todo
        seg = math.ceil(self.dim / ls)
        if hasattr(self, 'temp_g') == False:
            self.temp_g = create_metal_buffer_empty(self.n_heads*self.max_context*4,self.device)
        if hasattr(self, 'xq_temp_g') == False:
            self.xq_temp_g = create_metal_buffer_empty(self.dim*4,self.device)
        #threadgroup_barrier(mem_flags::mem_threadgroup);
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm(
            device float *a,
            device float *mean, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            int lidx0 = gid.x;
            float t = 0;
            for(int i = 0; i < {seg}; i++) {{
                t += a[lidx0*{seg} + i];
            }}
            temp[lidx0] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                mean[0] = t / {self.dim};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            t = 0;
            for(int i = 0; i < {seg}; i++) {{
                a[i + lidx0*{seg}] -= mean[0];
                t += pow(a[lidx0*{seg} + i],2);
            }}
            temp[lidx0] = t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                t = 0;
                for(int i = 0; i < {ls}; i++) {{
                    t += temp[i];
                }}
                mean[0] = pow(t / {self.dim} + 1e-5,0.5);
            }}
        }}

        kernel void mm4(
            device float *a,
            device const float *weight2, device const float *bias2,
            device const float *weight3, device const float *bias3,
            device float *mean,
            device float *h_temp, device float *h, device float *bias3_temp,
            uint3 gid [[thread_position_in_grid]])
        {{
            int lidx0 = gid.x % {ls3};
            int i = gid.x / {ls3};
            bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}] = bias3[i + lidx0*{math.ceil(self.dim*4 / ls3)}];
            for(int j = 0; j < {self.dim}; j++) {{
                bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}] += ((h[j] * weight2[j]) / mean[0] + bias2[j]) * weight3[(i + lidx0*{math.ceil(self.dim*4 / ls3)})*{self.dim} + j];
            }}
            float tth = bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}] * 0.7978845608\
            * (1 + 0.044715 * pow(bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}],2));
            float th = tanh(tth);
            if(isnan(th) && tth < 0) {{ th = -1;}}
            if(isnan(th) && tth >= 0) {{ th = 1;}}
            bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}] = 0.5 * bias3_temp[i + lidx0*{math.ceil(self.dim*4 / ls3)}]\
            * (1 + th);
        }}
        kernel void mm5(
            device float *a,
            device const float *weight4,device const float *bias4,
            device float *h_temp, device float *bias3_temp,
            uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float bias4_temp[{self.dim*3}];
            int lidx0 = gid.x % {ls3};
            int i = gid.x / {ls3};
            bias4_temp[lidx0 + i*{ls3}] = bias4[lidx0 + i*{ls3}];
            for(int j = 0; j < {self.dim*4}; j++) {{
                bias4_temp[lidx0 + i*{ls3}] += bias3_temp[j] * weight4[lidx0 + i*{ls3} + j*{self.dim}];
            }}
            a[lidx0 + i*{ls3}] = bias4_temp[lidx0 + i*{ls3}] + h_temp[lidx0 + i*{ls3}];
        }}
        """

        if prg_str not in self.prg_cache:
            library, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]

        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm1(
            device const float *a, device const float *c, device const float *d, device const float *e,
            device const float *xqkv, device float *keys_values,
            device float *xq_temp, device float *mean, uint3 gid [[thread_position_in_grid]])
        {{
            int lidx0 = gid.x % {ls};
            int i = gid.x / {ls};
            float t = 0;
            xq_temp[lidx0*{math.ceil(self.dim*3 / ls)} + i] = xqkv[lidx0*{int(self.dim*3 / ls)} + i];
            for(int k = 0; k < {self.dim}; k++) {{
                t += ((a[k] * c[k]) / mean[0] + d[k]) * e[(lidx0*{int(self.dim*3 / ls)} + i)*{self.dim} + k];
            }}
            if((lidx0*{int(self.dim*3 / ls)} + i) < {g}) {{
                xq_temp[lidx0*{int(self.dim*3 / ls)} + i] += t;
                }}
            if((lidx0*{int(self.dim*3 / ls)} + i) >= {g} && (lidx0*{int(self.dim*3 / ls)} + i) < {2*g}) {{
                keys_values[{start_pos}*{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {g}] = xqkv[{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {g}] + t;
            }}
            if((lidx0*{int(self.dim*3 / ls)} + i) >= {2*g}) {{
                keys_values[{self.dim*self.max_context} + {start_pos}*{self.dim} + lidx0*{int(self.dim*3 / ls)} + i - {2*g}] = xqkv[{self.dim*2} + lidx0*{int(self.dim*3 / ls)} + i - {2*g}] + t;
            }}
        }}
        kernel void mm2(
            device const float *keys_values, device float *temp3, device const float *xq_temp, uint3 gid [[thread_position_in_grid]])
        {{
            int lidx0 = gid.x;
            int x = (lidx0) % {start_pos+1};
            int k = (lidx0) / {start_pos+1};
            float acc0 = 0;
            for(int i = 0; i < 64; i++) {{
                acc0 += xq_temp[i + 64*k] * keys_values[x*{self.n_heads*64} + i + 64*k];
            }}          
            if(x + k*{start_pos+1} < {self.n_heads*self.max_context}) {{      
            temp3[x + k*{start_pos+1}] = acc0 / 8; //hardcoded math.sqrt(64)
            }}
        }}
        kernel void mm3(
            device float *a,
            device float *keys_values,
            device const float *weight,device const float *bias,
            device const float *weight2, device const float *bias2,
            device const float *weight3, device const float *bias3,
            device const float *weight4,
            device float *bias4,
            device float *temp3, device float *xq_temp, device float *mean,
            device float *h_temp, device float *h,
            uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls3}];
            int lidx0 = gid.x;
            if(lidx0 < {self.n_heads}){{
            float m = -INFINITY;
            for(int i = 0; i < {start_pos+1}; i++) {{
                m = max(m,temp3[i + lidx0*{start_pos+1}]);
            }}
            float t = 0;
            for(int i = 0; i < {start_pos+1}; i++) {{
                temp3[i + lidx0*{start_pos+1}] = exp(temp3[i + lidx0*{start_pos+1}] - m);
                t += temp3[i + lidx0*{start_pos+1}];
            }} //TODO SPLIT HERE???
            for(int i = 0; i < {start_pos+1}; i++) {{
                temp3[i + lidx0*{start_pos+1}] /= t;
            }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int g = 0; g < {seg3}; g++) {{ 
                float acc0 = 0;
                for(int i = 0; i < {start_pos+1}; i++) {{
                    acc0 += temp3[i + {start_pos+1}*((g + lidx0*{seg3}) / 64)] * keys_values[{self.dim*self.max_context} + i*{self.n_heads*64} + g + lidx0*{seg3}];
                }}
                xq_temp[g + lidx0*{seg3}] = acc0;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg3}; i++) {{
                float acc = 0;
                for(int x = 0; x < {self.dim}; x++) {{
                    acc += xq_temp[x] * weight[x*{self.dim} + lidx0*{seg3} + i];
                }}
                h[lidx0*{seg3} + i] = a[lidx0*{seg3} + i] + acc + bias[lidx0*{seg3} + i];
                h_temp[lidx0*{seg3} + i] = h[lidx0*{seg3} + i];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0;
            for(int i = 0; i < {seg3}; i++) {{
                total += h[lidx0*{seg3} + i];
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls3}; i++) {{
                    total += temp[i];
                }}
                mean[0] = total / {self.dim};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = 0;
            for(int i = 0; i < {seg3}; i++) {{
                h[i + lidx0*{seg3}] = h[i + lidx0*{seg3}] - mean[0];
                total += pow(h[lidx0*{seg3} + i],2);
            }}        
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0==0) {{
                total = 0;
                for(int i = 0; i < {ls3}; i++) {{
                    total += temp[i];
                }}
                mean[0] = pow(total / {self.dim} + 1e-5,0.5);
            }}
            }}
        """
        prg2, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)

        if hasattr(self, 'total') == False:
            self.total = create_metal_buffer_empty(1*4,self.device)

        if hasattr(self, 'bias3_temp') == False:
            self.bias3_temp = create_metal_buffer_empty(self.dim*4*4,self.device)
        if hasattr(self, 'mean') == False:
            self.mean = create_metal_buffer_empty(1*4,self.device)
        if hasattr(self, 'h_temp') == False:
            self.h_temp = create_metal_buffer_empty(self.dim*4,self.device)
        if hasattr(self, 'h') == False:
            self.h = create_metal_buffer_empty(self.dim*4,self.device)
        
        fxn = prg.newFunctionWithName_("mm")
        self.run_metal(fxn,1,256,[a_g\
        ,self.mean]) #TODO, reduce

        fxn = prg2.newFunctionWithName_("mm1")
        self.run_metal(fxn,math.ceil(self.dim*3 / ls),ls,[a_g,c_g,d_g,e_g,xqkv_g\
        ,keys_values_g,self.xq_temp_g,self.mean]) #TODO, reduce

        ls = 256
        fxn = prg2.newFunctionWithName_("mm2")
        def gs(ls): return math.ceil((self.n_heads*(start_pos+1)*(start_pos+1)) / ls)
        self.run_metal2(fxn,gs,ls,[keys_values_g ,self.temp_g, self.xq_temp_g])
        
        fxn = prg2.newFunctionWithName_("mm3")
        self.run_metal(fxn,1,ls3,[a_g\
        ,keys_values_g,weight_g,bias_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,weight4_g,bias4_g,self.temp_g, self.xq_temp_g,self.mean,
        self.h_temp,self.h])
        
        fxn = prg.newFunctionWithName_("mm4")
        self.run_metal(fxn,int(self.dim*4 / ls3),ls3,[a_g,\
        weight2_g,bias2_g,weight3_g,bias3_g,self.mean,
        self.h_temp,self.h,self.bias3_temp])

        fxn = prg.newFunctionWithName_("mm5")
        self.run_metal(fxn,int(self.dim / ls3),ls3,[a_g,\
        weight4_g,bias4_g,
        self.h_temp,self.bias3_temp])
        return a_g
 
    def kernel_2(self,x_g,ln_1_weight_g,ln_1_bias_g,attn_weight_g,attn_bias_g,cache_kv_g,attn_c_proj_weight_g,attn_c_proj_bias_g,ln_2_weight_g,ln_2_bias_g,c_fc_weight_g,c_fc_bias_g\
        ,c_proj_weight_g,c_proj_bias_g,num_tokens,max_content,j=0):
        if hasattr(self, 'h_g') == False:
            self.h_g = create_metal_buffer_empty(max_content*self.dim*4,self.device)
        if hasattr(self, 'h2_g') == False:
            self.h2_g = create_metal_buffer_empty(max_content*self.dim*4,self.device)
        if hasattr(self, 'xq_g') == False:
            self.xq_g = create_metal_buffer_empty(self.n_heads*64*max_content*4,self.device)
        if hasattr(self, 'xv_g') == False:
            self.xv_g = create_metal_buffer_empty(self.n_heads*64*max_content*4,self.device)
        if hasattr(self, 'c_g') == False:
            self.c_g = create_metal_buffer_empty(self.n_heads*64*max_content*4,self.device)
        if hasattr(self, 'xqt_g') == False:
            self.xqt_g = create_metal_buffer_empty(self.n_heads*64*max_content*4,self.device)
        if hasattr(self, 'res_g') == False:
            self.res_g = create_metal_buffer_empty(max_content*self.n_heads*4,self.device)
        if hasattr(self, 'xqkv_g') == False:
            self.xqkv_g = create_metal_buffer_empty(max_content*self.dim*3*4,self.device)
        if hasattr(self, 'd_g') == False:
            self.d_g = create_metal_buffer_empty(max_content*self.dim*4*4,self.device)
        a_rows = num_tokens
        a_cols = 64
        b_rows = self.dim
        ls = 256
        size = self.dim
        seg = int(size / ls) #todo
        b_cols = self.dim*3 # for first part
        b_cols_2 = self.dim*4
        prg_str = f"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void mm(
            device float *x, device const float *weight, device const float *bias,
            device float *copy, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            threadgroup float temp2[{num_tokens}];
            int gidx0 = gid.x;
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls};
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                copy[{self.dim}*r + lidx0*{seg} + i] = x[{self.dim}*r + lidx0*{seg} + i];
                temp2[r] += x[{self.dim}*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = temp2[r];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = temp2[lidx0] / {size};  
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] -= temp2[r];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            temp2[r] = 0;
            for(int i = 0; i < {seg}; i++) {{
                temp2[r] += pow(x[{self.dim}*r + lidx0*{seg} + i],2.0);
            }}
            temp[lidx0] = temp2[r];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{num_tokens}) {{
                temp2[lidx0] = 0;
                for(int i = 0; i < {ls}; i++) {{
                    temp2[lidx0] += temp[i];
                }}
                temp2[lidx0] = pow(temp2[lidx0] / {size} + 1e-5,0.5);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * weight[i + lidx0*{seg}]) / temp2[r] + bias[i + lidx0*{seg}];
                if(isnan(x[{self.dim}*r + i + lidx0*{seg}])) {{ x[{self.dim}*r + i + lidx0*{seg}] = 0; }} //TODO shouldn't need this
            }}
        }}
        kernel void mm2(
            device const float *x, device const float *attn_weight, device const float *attn_bias,device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            if(gidx0 < {b_cols*num_tokens}) {{
            int i = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += x[y*{b_rows} + k] * attn_weight[i*{b_rows} + k]; 
            }}
            res[y*{b_cols} + i] = total + attn_bias[i];
            }}
        }}
        kernel void mm3(
            device const float *xqkv, device float *new_cache, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = gidx0 / {self.n_heads*64};
            int j = gidx0 % {self.n_heads*64};
            new_cache[i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*1 + j];
            new_cache[{max_content}*{self.n_heads*64} + i*{self.n_heads*64} + j] = xqkv[i*{self.n_heads*64*3} + {self.dim}*2 + j]; 
        }}                 
        kernel void tr(
            device const float *xqkv, device float *xq, device float *xv,uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = (gidx0 / {64}) / {num_tokens};
            int j = (gidx0 / {64}) % {num_tokens};
            int k = gidx0 % 64;
            xq[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k];
            xv[i*{num_tokens}*64 + j*64 + k] = xqkv[i*64 + j*{64*self.n_heads*3} + k + {64*self.n_heads*2}];
        }}
        kernel void ms0(
            device float *xq, device const float *xqkv, uint3 gid [[thread_position_in_grid]])
        {{
            for(int gidx0 = 0; gidx0 < {self.n_heads*a_rows*a_rows}; gidx0++) {{
                int x = (gidx0 / {a_rows}) % {a_rows};
                int z = gidx0 / ({a_rows}*{a_rows}); 
                int y = gidx0 % {a_rows};
                float total = 0;
                for(int k = 0; k < {a_cols}; k++) {{
                    total += xq[y*{a_cols} + k + z*{a_rows}*{a_cols}] * xqkv[x*{64*self.n_heads*3} + k + z*64 + {self.dim}]; 
                }}
                xq[y*{a_rows} + x + z*{a_rows}*{a_rows}] = total / 8; //sqrt 64 input shape xq
            }}
        }}
        kernel void ms(
            device float *xq, uint3 gid [[thread_position_in_grid]])
        {{
        int gidx0 = gid.x;
        if(gidx0 < {self.n_heads*num_tokens*num_tokens}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = gidx0 % {num_tokens};
            if(z > y) {{ //todo, this can probably be 2x faster
                xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = -INFINITY;
            }}
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] = exp(xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z]);
        }}
        }}
        kernel void ms3(
            device const float *xq, device float *mx, uint3 gid [[thread_position_in_grid]])
        {{
        int gidx0 = gid.x;
        if(gidx0 < {num_tokens*self.n_heads}) {{
        int x = gidx0 / {num_tokens};
        int y = gidx0 % {num_tokens};
            float m = 0;
            for(int z = 0; z < {num_tokens}; z++) {{
                m += xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z];
            }}
            mx[x*{num_tokens} + y] = m;  
            }}
        }}
        kernel void ms4(
            device float *xq, device const float *mx, uint3 gid [[thread_position_in_grid]])
        {{
        int gidx0 = gid.x;
        if(gidx0 < {num_tokens*num_tokens*self.n_heads}) {{
            int x = (gidx0 / {num_tokens}) / {num_tokens};
            int y = (gidx0 / {num_tokens}) % {num_tokens};
            int z = gidx0 % {num_tokens};
            xq[x*{num_tokens}*{num_tokens} + y*{num_tokens} + z] /= mx[x*{num_tokens} + y];
        }}
        }}
        kernel void ms5(
            device const float *xq, device const float *xv, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int z = (gidx0 / {num_tokens}) / {a_cols};
            int x = (gidx0 / {num_tokens}) % {a_cols};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {num_tokens}; k++) {{
                total += xq[y*{num_tokens} + k + z*{num_tokens}*{num_tokens}] * xv[x + k*{a_cols} + z*{num_tokens}*{a_cols}]; 
            }}
            res[y*{a_cols} + x + z*{a_cols}*{num_tokens}] = total;
        }}
        kernel void ms6( //transpose
            device const float *xq, device float *xqt, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int i = (gidx0 / 64) / {num_tokens};
            int j = (gidx0 / 64) % {num_tokens};
            int k = gidx0 % 64;
            xqt[i*64 + j*{self.n_heads*64} + k] = xq[i*{num_tokens}*64 + j*64 + k];
        }}
        kernel void ms7(
            device const float *xq, device const float *attn_weight,device const float *attn_bias, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < {b_rows*num_tokens}) {{ //TODO don't allow larger local size? wasteful
            int gidx0 = gid.x;
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += xq[y*{b_rows} + k] * attn_weight[x*{b_rows} + k]; 
            }}
            res[y*{b_rows} + x] += total + attn_bias[x];
            }}
        }}
        
        kernel void ms8( //TODO, there are other kernels like this to fix
            device float *x, device const float *ln_2_weight, device const float *ln_2_bias
            ,device float *copy, uint3 gid [[thread_position_in_grid]])
        {{
            threadgroup float temp[{ls}];
            threadgroup float total;
            int gidx0 = gid.x;
            int lidx0 = gidx0 % {ls};
            int r = gidx0 / {ls}; //todo clean
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                copy[{self.dim}*r + lidx0*{seg} + i] = x[{self.dim}*r + lidx0*{seg} + i];
                total += x[{self.dim}*r + lidx0*{seg} + i];
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{num_tokens}) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] -= total / {size};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = 0;
            for(int i = 0; i < {seg}; i++) {{
                total += pow(x[{self.dim}*r + lidx0*{seg} + i],2);
            }}
            temp[lidx0] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lidx0<{num_tokens}) {{
                total = 0;
                for(int i = 0; i < {ls}; i++) {{
                    total += temp[i];
                }}
                total = pow(total / {size} + 1e-5,0.5);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i = 0; i < {seg}; i++) {{
                x[{self.dim}*r + i + lidx0*{seg}] = (x[{self.dim}*r + i + lidx0*{seg}] * ln_2_weight[i + lidx0*{seg}]) / total + ln_2_bias[i + lidx0*{seg}];
            }}
        }}

        kernel void ms9(
            device const float *a, device const float *c_fc_weight,device const float *c_fc_bias, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            int gidx0 = gid.x;
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_rows}; k++) {{
                total += a[y*{b_rows} + k] * c_fc_weight[x*{b_rows} + k];  //TODO A LEADS TO NANs
            }}
            float tth = (total + c_fc_bias[x]) * 0.7978845608\
                * (1 + 0.044715 * pow((total + c_fc_bias[x]),2));
            float th = tanh(tth);
            if(isnan(th) && tth < 0) {{
                th = -1;
            }}
            if(isnan(th) && tth >= 0) {{
                th = 1;
            }}
            res[y*{b_cols_2} + x] = 0.5 * (total + c_fc_bias[x])\
                * (1 + th);
        }}
        kernel void ms10(
            device const float *a, device const float *c_proj_weight,device const float *c_proj_bias, device float *res, uint3 gid [[thread_position_in_grid]])
        {{
            if(gid.x < {b_rows*num_tokens}) {{ //TODO, wasteful?
            int gidx0 = gid.x;
            int x = gidx0 / {num_tokens};
            int y = gidx0 % {num_tokens};
            float total = 0;
            for(int k = 0; k < {b_cols_2}; k++) {{
                total += a[y*{b_cols_2} + k] * c_proj_weight[x*{b_cols_2} + k];
            }}
            res[y*{b_rows} + x] += total + c_proj_bias[x];
            }}
        }}
        """
        if prg_str not in self.prg_cache:
            library, err = self.device.newLibraryWithSource_options_error_(prg_str, Metal.MTLCompileOptions.alloc().init(), None)
            self.prg_cache[prg_str] = library
        prg = self.prg_cache[prg_str]

        fxn = prg.newFunctionWithName_("mm")
        def gs(ls=None): return num_tokens     
        self.run_metal2(fxn,gs,ls,[x_g,ln_1_weight_g,ln_1_bias_g,self.h_g]) #TODO
        
        fxn = prg.newFunctionWithName_("mm2")
        def gs(ls): return math.ceil(b_cols*num_tokens / ls)
        self.run_metal2(fxn,gs,ls3,[x_g,attn_weight_g,attn_bias_g,self.xqkv_g])
       
        fxn = prg.newFunctionWithName_("mm3")
        def gs(ls): return math.ceil((num_tokens*self.n_heads*64) / ls)
        self.run_metal2(fxn,gs,ls3,[self.xqkv_g, cache_kv_g])
        
        fxn = prg.newFunctionWithName_("tr")
        def gs(ls): return math.ceil((num_tokens*self.n_heads*64) / ls)
        self.run_metal2(fxn,gs,ls3,[self.xqkv_g, self.xq_g, self.xv_g])

        fxn = prg.newFunctionWithName_("ms0")
        def gs(ls=None): return 1        
        self.run_metal2(fxn,gs,1,[self.xq_g, self.xqkv_g]) #TODO this is stable now, but slow

        fxn = prg.newFunctionWithName_("ms") 
        def gs(ls): return math.ceil(self.n_heads*num_tokens*num_tokens / ls)       
        self.run_metal2(fxn,gs,ls3,[self.xq_g])

        if self.n_heads*num_tokens > ls:
            g2 = math.ceil(self.n_heads*num_tokens / ls)*ls
        else:
            g2 = math.ceil(self.n_heads*num_tokens)
        g2 = math.ceil(self.n_heads*num_tokens)

        fxn = prg.newFunctionWithName_("ms3")
        def gs(ls): return 1        
        self.run_metal2(fxn,gs,self.n_heads*num_tokens,[self.xq_g,self.res_g])

        fxn = prg.newFunctionWithName_("ms4")
        def gs(ls): return math.ceil(self.n_heads*num_tokens*num_tokens / ls)  
        self.run_metal2(fxn,gs,ls3,[self.xq_g,self.res_g])
        
        fxn = prg.newFunctionWithName_("ms5")
        def gs(ls): return math.ceil(self.n_heads*a_cols*num_tokens / ls)
        self.run_metal2(fxn,gs,ls3,[self.xq_g,self.xv_g,self.c_g])
    
        fxn = prg.newFunctionWithName_("ms6")
        def gs(ls): return math.ceil(num_tokens*self.n_heads*64 / ls)
        self.run_metal2(fxn,gs,ls,[self.c_g,self.xqt_g])
            
        fxn = prg.newFunctionWithName_("ms7")
        def gs(ls): return math.ceil(b_rows*num_tokens / ls)
        self.run_metal2(fxn,gs,ls3,[self.xqt_g,attn_c_proj_weight_g,attn_c_proj_bias_g,self.h_g])
        
        fxn = prg.newFunctionWithName_("ms8")
        self.run_metal(fxn,num_tokens,ls,[self.h_g, ln_2_weight_g, ln_2_bias_g,self.h2_g]) #TODO fix
        
        fxn = prg.newFunctionWithName_("ms9")
        def gs(ls): return math.ceil(b_cols_2*num_tokens / ls)
        self.run_metal2(fxn,gs,ls3,[self.h_g, c_fc_weight_g,c_fc_bias_g,self.d_g])
                
        fxn = prg.newFunctionWithName_("ms10")
        def gs(ls): return math.ceil(b_rows*num_tokens / ls)        
        self.run_metal2(fxn,gs,ls3,[self.d_g, c_proj_weight_g,c_proj_bias_g,self.h2_g])
        return self.h2_g   

    def time_it(func,a,b,i=100):
        f = None
        total_time = 0
        for _ in range(i):
            st = time.perf_counter()
            ret = func(a,b)
            t = time.perf_counter() - st
            total_time += t
            if f is None or t < f:
                f = t
        return ret,f

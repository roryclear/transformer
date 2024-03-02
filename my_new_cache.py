from tinygrad import Tensor
from tinygrad.shape.symbolic import Variable
import numpy as np
import math

# can we get new_cache like gpt2.py with np??????
# makes output larger, but doesn't seem to change existing output?
MAX_CONTEXT=4 #128 normally
seqlen = 1 #length of prompt, then 1 after (1 token inputted)

#find out about this bind, number is pos of token in text
#BIND VALUE MAKES NO DIFFERENCE??, but is still needed
start_pos = Variable("start_pos",1,MAX_CONTEXT).bind(3)

Tensor.manual_seed(420)

#keys = Tensor.rand(1, MAX_CONTEXT, 12, 64)
#values = Tensor.rand(1, MAX_CONTEXT, 12, 64)
#xv = Tensor.rand(1, 1, 12, 64)
#xk = Tensor.rand(1, 1, 12, 64)

#chanding these also doesn't seem to change output, only adds to them
#keys = Tensor.empty(1, MAX_CONTEXT, 3, 2)
#keys = Tensor.full_like(keys,2.0)
def tg_new_cache(keys,values,xv,xk,start_pos):
    keystg = keys.shrink((None, (0, start_pos), None, None))
    keystg = keystg.cat(xk, dim=1)
    valuestg = values.shrink((None, (0, start_pos), None, None)).cat(xv, dim=1)
    new_cache = Tensor.stack([keystg, valuestg]).pad((None, None,(0,MAX_CONTEXT-start_pos-seqlen),None,None)).contiguous()
    return new_cache.numpy()

def my_new_cache(keys,values,xv,xk,start_pos):
    keys = keys.numpy()
    values = values.numpy()
    xv = xv.numpy()
    xk = xk.numpy()
    #mess todo later
    s = list(np.shape(keys))
    s[1] = 4
    keys_small = np.zeros(shape=s)
    values_small = np.zeros(shape=s)
    keys_small[0] = keys[0][0:4]
    values_small[0] = values[0][0:4]
    ret = [keys_small,values_small]
    #terrible loop
    for a in range(len(ret)):
        for b in range(len(ret[0])):
            for c in range(start_pos.unbind()[1]+1,len(ret[0][0])):
                ret[a][b][c] = np.zeros_like(ret[a][b][c])
        if start_pos.unbind()[1] > -1 and start_pos.unbind()[1] < len(ret[0][0]):
            ret[0][b][start_pos.unbind()[1]] = xk[0][0]
            ret[1][b][start_pos.unbind()[1]] = xv[0][0]
    return ret

def zero_tg_new_cache(keys,values,seqlen,start_pos=0): 
    ret = Tensor.stack([keys, values]).pad((None, None,(0,MAX_CONTEXT-start_pos-seqlen),None,None)).contiguous()
    return ret

def zero_my_new_cache(keys,values,seqlen):
    keys = keys.numpy()
    values = values.numpy()
    s = list(np.shape(keys))
    s[1] = MAX_CONTEXT
    ret = np.zeros(shape=s)
    ret = [np.copy(ret),np.copy(ret)]
    for i in range(len(keys[0])):
        ret[0][0][i] = keys[0][i]
        ret[1][0][i] = values[0][i]       
    return ret

######my attempt#######
start_pos = Variable("start_pos",1,4).bind(3)
keys = Tensor.ones(1,4,3,2)
values = Tensor.ones(1,4,3,2)
xv = Tensor.ones(1,1,3,2)
xk = Tensor.ones(1,1,3,2)
new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
my_cache = my_new_cache(keys,values,xv,xk,start_pos)
np.testing.assert_allclose(new_cache,my_cache)

for i in range(1,5):
    start_pos = Variable("start_pos",1,4).bind(i)
    new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
    my_cache = my_new_cache(keys,values,xv,xk,start_pos)
    np.testing.assert_allclose(new_cache,my_cache)


for i in range(1,5):
    start_pos = Variable("start_pos",1,4).bind(3)
    keys = Tensor.full_like(keys,5)
    new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
    my_cache = my_new_cache(keys,values,xv,xk,start_pos)
    np.testing.assert_allclose(new_cache,my_cache)

start_pos = Variable("start_pos",1,4).bind(3)
keys = Tensor.rand(keys.shape)
values = Tensor.rand(values.shape)
new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
my_cache = my_new_cache(keys,values,xv,xk,start_pos)
np.testing.assert_allclose(new_cache,my_cache)

start_pos = Variable("start_pos",1,4).bind(4)
# (1, 128, 12, 64)
xv = Tensor.rand(1,1,3,2)
xk = Tensor.rand(1,1,3,2)
keys = Tensor.rand(1,18,3,2)
values = Tensor.rand(1,18,3,2)
#keys = Tensor.full_like(keys,5)
#values = Tensor.full_like(values,4)
new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
my_cache = my_new_cache(keys,values,xv,xk,start_pos)
np.testing.assert_allclose(new_cache,my_cache)


## for case where start_pos <= 0
start_pos = 0 #always 0 and type int
# keys and values shapes are (1, 13, 12, 64) (13 is the length of the prompt)
# nothing else in the shape of keys, values and new_cache shapes are dependent on the prompt
# new_values shapes = (2, 1, 128, 12, 64)
# if this func it is (2, 1, 16, 12, 64)

MAX_CONTEXT=128

seqlen = 13 #length of the prompt !
keys = Tensor.ones(1,seqlen,3,4)
values = Tensor.ones(1,seqlen,3,4)
values = Tensor.full_like(values,2)
new_cache = zero_tg_new_cache(keys,values,seqlen,0)
my_new_cache = zero_my_new_cache(keys,values,seqlen)

np.testing.assert_allclose(new_cache.numpy(),my_new_cache)

# start pos = start_pos[1-MAX_CONTENT=pos] so [1-128=13] ...[1-128=111]
# keys and values both shape (1,128,12,64)
# xq shape is (1, 1, 12, 64)
def make_xq(keys,values,xq,xk,xv,start_pos):
    keys = keys.shrink((None, (0, start_pos), None, None))
    keys = keys.cat(xk, dim=1)
    values = values.shrink((None, (0, start_pos), None, None))
    values = values.cat(xv, dim=1)
    keys, values = keys.transpose(1, 2), values.transpose(1, 2)
    qk2 = xq @ keys.transpose(-2,-1) / math.sqrt(xq.shape[-1])
    xq = qk2.softmax(-1) @ values
    return xq

def my_make_xq(keys,values,xq,xk,xv,start_pos):
    xk = xk.numpy()
    xv = xv.numpy()
    values = values.numpy()
    ret = np.zeros((1,np.shape(xk)[2],np.shape(xk)[2],np.shape(xk)[3]))
    ret = np.array(ret)
    ret += xv / (start_pos.unbind()[1] + 1)
    ret = ret.transpose(0,2,1,3)
    
    for a in range(len(ret[0])):
        for b in range(len(ret[0][a])):
            for c in range(len(ret[0][a][b])):
                ret[0][a][b][c] += values[0][a][b][c]*(1-1/(start_pos.unbind()[1] + 1)) 
    return ret


MAX_CONTEXT = 128
start_pos = Variable("start_pos",1,MAX_CONTEXT).bind(14)
xk = Tensor.zeros((1,1,12,64))
xv = Tensor.zeros((1,1,12,64))
xq = Tensor.zeros((1,1,12,64))
keys = Tensor.zeros((1,128,12,64))
values = Tensor.zeros((1,128,12,64))
xq_out = make_xq(keys,values,xq,xk,xv,start_pos)
my_xq_out = my_make_xq(keys,values,xq,xk,xv,start_pos)
np.testing.assert_allclose(xq_out.numpy(),my_xq_out)

MAX_CONTEXT = 8
start_pos = Variable("start_pos",1,MAX_CONTEXT).bind(4)
xk = Tensor.zeros((1,1,4,5))
xv = Tensor.ones((1,1,4,5))
xq = Tensor.zeros((1,1,4,5))
keys = Tensor.zeros((1,MAX_CONTEXT,4,5))
values = Tensor.zeros((1,MAX_CONTEXT,4,5))
xq_out = make_xq(keys,values,xq,xk,xv,start_pos)
my_xq_out = my_make_xq(keys,values,xq,xk,xv,start_pos)
np.testing.assert_allclose(xq_out.numpy(),my_xq_out)

MAX_CONTEXT = 8
start_pos = Variable("start_pos",1,MAX_CONTEXT).bind(4)
xk = Tensor.zeros((1,1,4,5))
xv = Tensor.rand((1,1,4,5))
xq = Tensor.zeros((1,1,4,5))
keys = Tensor.zeros((1,MAX_CONTEXT,4,5))
values = Tensor.zeros((1,MAX_CONTEXT,4,5))
xq_out = make_xq(keys,values,xq,xk,xv,start_pos)
my_xq_out = my_make_xq(keys,values,xq,xk,xv,start_pos)
np.testing.assert_allclose(xq_out.numpy(),my_xq_out,atol=1e-6)

MAX_CONTEXT = 8
start_pos = Variable("start_pos",1,MAX_CONTEXT).bind(4)
xk = Tensor.zeros((1,1,4,5))
#xk = Tensor.full_like(xk,2)
xv = Tensor.zeros((1,1,4,5))
xq = Tensor.zeros((1,1,4,5))
keys = Tensor.zeros((1,MAX_CONTEXT,4,5))
values = Tensor.zeros((1,MAX_CONTEXT,4,5))
values = Tensor.full_like(values,2)
values = values.numpy()

values = Tensor(values)
xq_out = make_xq(keys,values,xq,xk,xv,start_pos)
my_xq_out = my_make_xq(keys,values,xq,xk,xv,start_pos)
np.testing.assert_allclose(xq_out.numpy(),my_xq_out,atol=1e-6)
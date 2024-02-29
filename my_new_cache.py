from tinygrad import Tensor
from tinygrad.shape.symbolic import Variable
import numpy as np

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
            ret[0][b][start_pos.unbind()[1]] = np.full_like(ret[a][b][0],xk[0][0][0][0])
            ret[1][b][start_pos.unbind()[1]] = np.full_like(ret[a][b][0],xv[0][0][0][0])
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

start_pos = Variable("start_pos",1,4).bind(3)
xv = Tensor.ones(1,1,3,2)
xv = Tensor.full_like(xv,6)
xk = Tensor.ones(1,1,3,2)
xk = Tensor.full_like(xk,7)
keys = Tensor.ones(1,20,3,2)
values = Tensor.ones(1,20,3,2)
keys = Tensor.full_like(keys,5)
values = Tensor.full_like(values,4)
new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
my_cache = my_new_cache(keys,values,xv,xk,start_pos)
#print(np.shape(new_cache))
print(new_cache,np.shape(new_cache))
print("MINE =\n")
print(my_cache,np.shape(new_cache))
#print(np.shape(my_cache))
np.testing.assert_allclose(new_cache,my_cache)

#actual shape is (1, 128, 12, 64)

'''
keys = Tensor.zeros(1,4,3,2)
keys = Tensor.full_like(keys,5)
start_pos = Variable("start_pos",1,4).bind(1)
new_cache = tg_new_cache(keys,values,xv,xk,start_pos)
my_cache = my_new_cache(keys,values,xv,xk,start_pos)
print("ANSWER = ",new_cache)
print("MY CACHE =",my_cache)
np.testing.assert_allclose(new_cache,my_cache)
'''
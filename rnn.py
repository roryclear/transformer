from tinygrad import nn, Tensor
from tinygrad.nn.optim import SGD
from typing import Tuple
import time
'''
class RNN:
    def __init__(self,hidden_size=128):
        self.hidden_size = hidden_size
        self.w_in = nn.Linear(vocab_size,hidden_size)
        self.h0 = nn.Linear(hidden_size*2,hidden_size)
        self.w_out = nn.Linear(hidden_size,vocab_size)
        self.prev_hidden = Tensor.zeros(hidden_size)
        self.hidden = Tensor.zeros(hidden_size)
    
    def __call__(self,x:Tensor):
        x = self.w_in(x)
        #print("x now =",x.numpy(),len(x.numpy()))
        self.hidden = x
        x = x.cat(self.prev_hidden)
        self.prev_hidden = self.hidden
        x = self.h0(x)
        x = self.w_out(x)
        return x
'''
class tinyrnn:
    def __init__(self,hidden_size=8):
        self.hidden_size = hidden_size
        self.prevh = Tensor.zeros(self.hidden_size)
        self.w_in = nn.Linear(8,8)
        self.h0 = nn.Linear(8+hidden_size,self.hidden_size)
        self.w_out = nn.Linear(self.hidden_size,8)
    
    def __call__(self, x):
        ret = self.w_in(x)
        ret = ret.cat(self.prevh)
        ret = Tensor.tanh(ret)
        ret = self.h0(ret)
        self.prevh = ret
        ret = self.w_out(ret)
        ret = ret.softmax()
        return ret

model = tinyrnn(hidden_size=8) #8 is more than enough for "roryclear." overfit
# depends on the init, can we use a seed?
input = "roryclear."
chars = list(set(input)) #['r','o','y','c','l','e','a','.']
chars = ['r','o','y','c','l','e','a','.']
vocab_size = len(chars)

def str_to_tensor(s):
    ret = Tensor(None)
    for c in s:
        print(c,chars.index(c))
        c_tensor = Tensor([1])
        c_tensor = c_tensor.pad2d([chars.index(c),vocab_size-chars.index(c)-1],0)
        ret = ret.cat(c_tensor)
        print("c_tensor =",c_tensor.numpy())
    ret = ret.reshape(len(s),vocab_size)
    print("ret =",ret.numpy())
    return ret
inputTensor  = str_to_tensor("roryclear")
print("inputTensor ->",inputTensor)
''' function should make this now
inputTensor = Tensor([[1,0,0,0,0,0,0,0], #r
                      [0,1,0,0,0,0,0,0], #o
                      [1,0,0,0,0,0,0,0], #r
                      [0,0,1,0,0,0,0,0], #y
                      [0,0,0,1,0,0,0,0], #c
                      [0,0,0,0,1,0,0,0], #l
                      [0,0,0,0,0,1,0,0], #e
                      [0,0,0,0,0,0,1,0], #a
                      [1,0,0,0,0,0,0,0], #r
                      ])
'''
targetTensor = Tensor([[1],[0],[2],[3],[4],[5],[6],[0],[7]]) # o r y c l e a r .

opt = nn.optim.Adam([model.w_in.weight,model.h0.weight,model.w_out.weight], lr=1e-3)
for e in range(10000):
    model.prevh = Tensor.zeros(model.hidden_size)
    for i in range(inputTensor.shape[0]):
        opt.zero_grad()
        out = model(inputTensor[i])
        loss = out.sparse_categorical_crossentropy(targetTensor[i])
        loss.backward()
        opt.step()
        if i == 6:
            print("loss =",loss.numpy())
    
    if e % 100 == 0:
        s = chars[inputTensor[0].argmax().numpy()]
        print("epoch",e)
        for i in range(inputTensor.shape[0]):
            out = model(inputTensor[i])
            s += chars[out.argmax().numpy()]
        print("output =",s)
        if s == "roryclear.":
            print("CORRECT")
            exit()

        

exit()
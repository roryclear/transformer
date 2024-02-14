from tinygrad import nn, Tensor
from tinygrad.nn.optim import SGD
from typing import Tuple
import time
import random
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
    def __init__(self,hidden_size=8,vocab_size=8):
        self.hidden_size = hidden_size
        self.prevh = Tensor.zeros(self.hidden_size)
        self.w_in = nn.Linear(vocab_size,8)
        self.h0 = nn.Linear(8+hidden_size,self.hidden_size)
        self.w_out = nn.Linear(self.hidden_size,vocab_size)
    
    def __call__(self, x):
        ret = self.w_in(x)
        ret = ret.cat(self.prevh)
        ret = Tensor.tanh(ret)
        ret = self.h0(ret)
        self.prevh = ret
        ret = self.w_out(ret)
        ret = ret.softmax()
        return ret
    
def str_to_input_tensor(s):
    ret = Tensor(None)
    for c in s:
        c_tensor = Tensor([1]).pad2d([chars.index(c),vocab_size-chars.index(c)-1],0)
        ret = ret.cat(c_tensor)
    return ret.reshape(len(s),vocab_size)

def str_to_target_tensor(s):
    ret = Tensor(None)
    for c in s:
        ret = ret.cat(Tensor([chars.index(c)]))
    ret = ret.reshape(len(s),1)
    return ret
'''
model = tinyrnn(hidden_size=8,vocab_size=27)

lines = open('data/names.txt', 'r').readlines()
lines = list(set(lines)) #unique lines only!
chars = "."
for i in range(len(lines)): 
    lines[i] = lines[i].replace("\n","")
    chars += lines[i]
chars = list(set(chars))
vocab_size = len(chars)
lines.sort()
random.Random(420).shuffle(lines) #same shuffle every time
train_names = lines[:int(len(lines)*0.9)]
test_names = lines[int(len(lines)*0.9):]
print("first =",train_names[0],test_names[0])
all = ""
for i in range(len(train_names)):
    print(i,train_names[i])
    input = str_to_input_tensor(train_names[i])
    target = str_to_target_tensor(train_names[i][1]+".")
    print("input =",input.numpy())
    print("target =",target.numpy(),"\n")
    out = model(input)
    loss = out.sparse_categorical_crossentropy(target)

exit()
'''


model = tinyrnn(hidden_size=8) #8 is more than enough for "roryclear." overfit
# depends on the init, can we use a seed?
input = "roryclear."
chars = list(set(input)) #['r','o','y','c','l','e','a','.']
chars = ['r','o','y','c','l','e','a','.']
vocab_size = len(chars)

input_tensor  = str_to_input_tensor("roryclear")
target_tensor = str_to_target_tensor("oryclear.") #Tensor([[1],[0],[2],[3],[4],[5],[6],[0],[7]])
# o r y c l e a r .

opt = nn.optim.Adam([model.w_in.weight,model.h0.weight,model.w_out.weight], lr=1e-2)
for e in range(10000):
    model.prevh = Tensor.zeros(model.hidden_size)
    for i in range(input_tensor.shape[0]):
        out = model(input_tensor[i])
        loss = out.sparse_categorical_crossentropy(target_tensor[i])
        loss.backward()
    opt.step() # slower for "roryclear." overfit example obvs
    opt.zero_grad()
    print("loss =",loss.numpy())
    
    if e % 100 == 0:
        s = chars[input_tensor[0].argmax().numpy()]
        print("epoch",e)
        for i in range(input_tensor.shape[0]):
            out = model(input_tensor[i])
            s += chars[out.argmax().numpy()]
        print("output =",s)
        if s == "roryclear.":
            print("CORRECT")
            break
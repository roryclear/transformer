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
    def __init__(self):
        self.prevh = Tensor.zeros(8)
        self.w_in = nn.Linear(8,8)
        self.w_out = nn.Linear(8,8)
    
    def __call__(self, x):
        ret = self.w_in(x)
        ret = self.w_out(ret)
        return ret

model = tinyrnn()
input = "roryclear."
chars = ['r','o','y','c','l','e','a','.']
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

targetTensor = Tensor([[1],[0],[2],[3],[4],[5],[6],[0],[7]]) # o r y c l e a r .

opt = nn.optim.Adam([model.w_in.weight,model.w_out.weight], lr=1e-3)
for e in range(1000):
    for i in range(inputTensor.shape[0]):
        opt.zero_grad()
        out = model(inputTensor[i])
        loss = out.sparse_categorical_crossentropy(targetTensor[i])
        loss.backward()
        opt.step()
        if i == 6:
            print("rory loss =",loss.numpy())
    
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
# r = 0, o = 1, y = 2, c = 3, l = 4, e = 5, a = 6
r = Tensor([1,0,0,0,0,0,0])
o = Tensor([0,1,0,0,0,0,0])
opt = nn.optim.Adam([model.w_in.weight,model.w_out.weight], lr=1e-3)
opt.zero_grad()
batch = Tensor([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0]]) #r o, to try out?
for i in range(1000):
    out = model(r)
    loss = out.sparse_categorical_crossentropy(Tensor([1])) # just use the category positions in an array?? o r
    loss.backward()
    opt.step()
    print("rory loss =",loss.numpy())
exit()

class RNN:
    def __init__(self,hidden_size=128):
        #self.hidden_size = hidden_size
        self.w_in = nn.Linear(vocab_size,hidden_size)
        self.h0 = nn.Linear(hidden_size*2,hidden_size)
        self.w_out = nn.Linear(hidden_size,vocab_size)
        self.prev_hidden = Tensor.zeros(hidden_size)
        self.hidden = Tensor.zeros(hidden_size)
    
    def __call__(self,x:Tensor):
        x = self.w_in(x)
        #print("x now =",x.numpy(),len(x.numpy()))
        x = x.cat(self.prev_hidden)
        x = self.h0(x)
        x = Tensor.tanh(x)
        self.prev_hidden = x
        x = self.w_out(x)
        return x

print("rnn??")

names = open('data/names.txt', 'r').readlines()
print(len(names),"names in file")

chars = set()
unique = set() #some names like rory are there twice
for i in range(len(names)):
    names[i] = names[i].replace("\n",".")
    if names[i] not in unique:
        unique.add(names[i])
    for y in names[i]:
        if y not in chars:
            chars.add(y)
names = list(unique)
print(len(names),"names now")
chars = sorted(list(chars))
print(chars)
vocab_size = len(chars) #"special 0 token?"-karpathy
i2s = {}
s2i = {}
# "decode and encode"
for i in range(len(chars)):
    i2s[i] = chars[i]
    s2i[chars[i]] = i

  
#model2 = ActorCritic(10, int(10))    # type: ignore
#opt = nn.optim.Adam(nn.state.get_parameters(model2), lr=3e-4)

model = RNN()
#opt = nn.optim.Adam(nn.state.get_parameters(model), lr=3e-4)
opt = nn.optim.Adam([model.w_in.weight,model.h0.weight,model.w_out.weight], lr=3e-4)
#print(nn.state.get_parameters(model))
#exit()
#exit()
x = 0
hidden_size = 128
total_loss = Tensor(0)
avg_acc = 0
acc = 0
w = 0
st = time.perf_counter()
for n in names:
    w+=1
    #print(w)    
    model.prev_hidden = Tensor.zeros(hidden_size)
    for i in range(0,len(n)-2,2):
        #print(n,":",n[i]," ->",n[i+1])
        e = s2i[n[i]]
        e2 = s2i[n[i+1]]
        input = Tensor([1]).pad2d([e,vocab_size-e-1])
        print("rory input shape =",input.shape)
        input = input.cat(Tensor([1]).pad2d([e2,vocab_size-e2-1]))
        input = input.reshape(2,27)
        #print("input =",input.numpy(),"len =",len(input.numpy()))
        opt.zero_grad()

        out = model(input)
        #print("output =",out.numpy())

        loss = out.sparse_categorical_crossentropy(Tensor(s2i[n[i+1]]))

        pred = out.argmax(axis=-1)
        #acc += (pred == s2i[n[i+1]]).mean().numpy()
        #print(w,"\t",avg_acc,"\tletter =",n,(n[:i+1]+i2s[int(pred.numpy())])) #just do this for test?
        x+=1
        
        loss.backward()
        opt.step()

        #timing first 5000 so can improve it?
        if x > 500:
            t = time.perf_counter() - st
            print("time taken =",t)
            exit()

    if w > 0 and w % 100 == 0:
        avg_acc = acc/x
        print(w,"acc =",avg_acc)
        acc = 0
        x = 0
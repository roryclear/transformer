from tinygrad import nn, Tensor
from tinygrad.nn.optim import SGD
from typing import Tuple
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
for i in range(len(names)):
    names[i] = names[i].replace("\n",".")
    for y in names[i]:
        if y not in chars:
            chars.add(y)
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
for n in names:
    w+=1
    #print(w)    
    model.prev_hidden = Tensor.zeros(hidden_size)
    for i in range(len(n)-1):
        #print(n,":",n[i]," ->",n[i+1])
        e = s2i[n[i]]
        #print(n[i],e)
        input = Tensor([1]).pad2d([e,vocab_size-e-1])
        #print("input =",input.numpy(),"len =",len(input.numpy()))
        opt.zero_grad()

        out = model(input)
        #print("output =",out.numpy())

        loss = out.sparse_categorical_crossentropy(Tensor(s2i[n[i+1]]))

        pred = out.argmax(axis=-1)
        acc += (pred == s2i[n[i+1]]).mean().numpy()
        print(w,"\t",avg_acc,"\tletter =",n,(n[:i+1]+i2s[int(pred.numpy())])) #just do this for test?
        x+=1
        #print("rory acc =",acc.numpy())

        #print("rory loss =",loss.numpy())
        loss.backward()

        opt.step()
        #print("rory loss =",loss.numpy())
    if w > 0 and w % 100 == 0:
        avg_acc = acc/x
        print(w,"acc =",avg_acc)
        acc = 0
        x = 0
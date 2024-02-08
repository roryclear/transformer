from tinygrad import nn, Tensor

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
        print("x now =",x.numpy(),len(x.numpy()))
        self.hidden = x
        x = x.pad2d([0,self.hidden_size],0)
        print("x now =",x.numpy(),len(x.numpy()))
        x = x + self.prev_hidden.pad2d([self.hidden_size,0])
        print("x now =",x.numpy(),len(x.numpy()))
        self.prev_hidden = self.hidden
        x = self.h0(x)
        x = self.w_out(x)
        return x

print("rnn??")

names = open('data/names.txt', 'r').readlines()
print(len(names),"names in file")

chars = set()
for i in range(len(names)):
    names[i] = names[i].replace("\n","")
    for y in names[i]:
        if y not in chars:
            chars.add(y)
chars = sorted(list(chars))
print(chars)
vocab_size = len(chars) + 1 #"special 0 token?"-karpathy
i2s = {}
s2i = {}
# "decode and encode"
for i in range(len(chars)):
    i2s[i] = chars[i]
    s2i[chars[i]] = i

model = RNN()

for n in names[0:2]:
    for i in range(len(n)-1):
        print(n,":",n[i]," ->",n[i+1])
        e = s2i[n[i]]
        print(n[i],e)
        input = Tensor([1]).pad2d([e,vocab_size-e-1])
        print("input =",input.numpy(),"len =",len(input.numpy()))
        out = model(input)
        print("output =",out.numpy())

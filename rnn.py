print("rnn??")

names = open('data/names.txt', 'r').readlines()
print(len(names),"names in file")

chars = set()
for x in names:
    for y in x:
        if y not in chars:
            chars.add(y)
chars.remove("\n")
chars = sorted(list(chars))
print(chars)
vocab_size = len(chars) + 1 #"special 0 token?"-karpathy
i2s = {}
s2i = {}
# "decode and encode"
for i in range(len(chars)):
    i2s[i] = chars[i]
    s2i[chars[i]] = i

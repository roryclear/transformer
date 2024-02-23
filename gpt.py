import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

def make_token_file():
    f = open("tokens.txt", "w")
    for i in range(50257):
        print(i,tokenizer.decode([i]))
        if "/n" in tokenizer.decode([i]):
            exit()
        f.write(tokenizer.decode([i]).replace("\n","/n")+"\n")
    f.close()

make_token_file()
tokens = open('tokens.txt', 'r').readlines()
for i in range(50257):
    print(i,tokenizer.decode([i]),tokens[i].replace("\n","\n"))
    
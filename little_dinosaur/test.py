import fairies as fa

a = fa.read_npy('output.npy')

# print(a)

for i in a:
    print(i)

print(len(a))
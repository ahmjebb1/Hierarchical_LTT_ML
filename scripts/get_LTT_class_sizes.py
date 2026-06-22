import pickle
import os

# need to modify this path
directory = '/users/li1ajs/fastdata/LTT'
# to check it worked:
#directory = '/users/li1ajs/fastdata/WWr'

sizes=[]
for filename in os.listdir(directory):
  if filename.endswith('.pkl'):
    with open(os.path.join(directory,filename), 'rb') as f:
      data = pickle.load(f)
      sizes.append(len(data))
      print(f"{filename},{len(data)}")

print(sizes)

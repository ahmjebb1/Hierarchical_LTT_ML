import os
import pickle
import random

# List of the class sizes of each of the LTT classes
class_sizes=[9, 12, 8, 16, 30, 4, 43, 48, 32, 8, 10, 16, 30, 10, 29, 29, 10, 46, 34, 22, 103, 18, 6, 19, 20, 11, 23, 25, 1, 38, 6, 42, 10, 43, 9, 27, 12, 2, 32]

# Post-March 2025 list of the class sizes:
class_sizes=[62, 122, 7, 75, 60, 57, 41, 8, 67, 1, 123, 70, 2, 55, 134, 17, 52, 117, 29, 298, 75, 68, 89, 68, 63, 8, 62, 146, 16, 51, 56, 40, 56, 12, 3, 85, 4, 62, 45, 170]

# Source of the (larger) data set
source_dir="/users/li1ajs/fastdata/WWc"

# Where to put the smaller data set
dest_dir="/users/li1ajs/fastdata/WWr"

source_class_sizes=[]

# Get a list of the source data set's class sizez
for filename in os.listdir(source_dir):
 with open(source_dir+"/"+filename, 'rb') as f:
  birdcalls = pickle.load(f)
  print(f"{filename}, {len(birdcalls)}")
  source_class_sizes.append((filename,len(birdcalls)))

# Sort the source and target datasets into largest first:
sorted_target = sorted(class_sizes, reverse=True)
sorted_source = sorted(source_class_sizes, key=lambda x: x[1], reverse=True)

print(sorted_source)
print(sorted_target)

# For every target class, create a source class of the same size:
for t in sorted_target:
 # pop the first (largest) from the source:
 largest = sorted_source.pop(0)

 # load the pickle, truncate randomly to size t:
 filename = largest[0]
 size = largest[1]
 if size < t: print("Warning, a source item is smaller than the target!")
 with open(source_dir+"/"+filename, 'rb') as f:
  birdcalls = pickle.load(f)
 print(f"{filename}, {len(birdcalls)}")
 truncated = random.sample(birdcalls, t)
 
 # save to destination dir
 with open(dest_dir+"/"+filename, 'wb') as g:
  pickle.dump(truncated, g)
 

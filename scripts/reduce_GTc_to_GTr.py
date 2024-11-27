import os
import pickle
import random

# List of the class sizes of each of the LTT classes
class_sizes=[9, 12, 8, 16, 30, 4, 43, 48, 32, 8, 10, 16, 30, 10, 29, 29, 10, 46, 34, 22, 103, 18, 6, 19, 20, 11, 23, 25, 1, 38, 6, 42, 10, 43, 9, 27, 12, 2, 32]

# Source of the (larger) data set
source_dir="data_classes/0"

# Where to put the smaller data set
dest_dir="gt_data_ltt_shape"

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
 

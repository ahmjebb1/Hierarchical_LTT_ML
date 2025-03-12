"""
Reads all the .pkl files and sorts them to group data according to labelled bird and to remove duplicates
"""
import pickle
from pathlib import Path
import numpy as np
import re, os, sys, shutil

def bytes_to_spectrogram(bytes_in):
    array_1d = np.frombuffer(bytes_in, np.float32)
    return np.reshape(array_1d, (128, len(array_1d)//128))

# Settings/argparse
if len(sys.argv) != 3:
    print("Two arguments required")
    print("Usage: python %s <in_directory> <out_directory>"%(sys.argv[0]))
    sys.exit(0)
_in_dir = sys.argv[1]
_out_dir = sys.argv[2]
if not os.path.exists(_in_dir):
    print("Error: input directory '%s' does not exist."%(_in_dir))
    sys.exit(1)
if os.path.exists(_out_dir):
    if os.listdir(_out_dir) != []:
        print("Output directory '%s' is not empty."%(_out_dir))
        print("Continuing will cause the directory to be overwritten.")
        print("Continue? y/n")
        response = input().lower()
        while response != "y" and response != "n":
            print("Continue? y/n")
            response = input().lower()
        if response == "n":
            sys.exit(1)
        elif response == "y":
            shutil.rmtree(_out_dir)

# Dictionary for loading all pickle data
loaded_data = {}
total_loaded_items = 0

# Regex string to capture birds ID and strip number from label
re_cleanlabel = r"(.+)_[0-9]+"

# Iterate and load all .pkl files within _in_dir, removing dupes
for label_path in Path(_in_dir).rglob('*_labels.pkl'):
    # Data is split across two separate pickles, data and labels so must be loaded in pairs
    with open(label_path, "rb") as pkl_file:
        labels = pickle.load(pkl_file)
    
    spectrogram_path = Path(str(label_path).replace("_labels", ""))
    if spectrogram_path.is_file():
        with open(spectrogram_path, "rb") as pkl_file:
            spectrograms = pickle.load(pkl_file)
    else:
        print("Warning: Label .pkl '%s' does not have a corresponding spectrogram_path and was skipped."%(label_path.relative_to(_in_dir)))
        continue
        
    total_loaded_items += len(labels)
    for i in range(len(labels)):
        # Strip the index from the label, so that we just have bird ID
        clean_label = re.search(re_cleanlabel, labels[i]).groups(1)[0] # returns a tuple for some reason, we just want the first element
        # Store the associated spectrogram for that bird (as bytes)
        if clean_label not in loaded_data:
            loaded_data[clean_label] = set()
        spectrogram_bytes = spectrograms[i].data.tobytes()
        loaded_data[clean_label].add(spectrogram_bytes)
        # Process data back from bytes to validate data is retained correctly
        if not np.array_equal(spectrograms[i], bytes_to_spectrogram(spectrogram_bytes)):
            print("Warning: Failed to convert spectrogram back from bytes correctly.")

# For each set within loaded_data, replace the set with a list of spectograms by converting bytes back to np.array
for k in loaded_data.keys():
    loaded_data[k] = list([bytes_to_spectrogram(i) for i in loaded_data[k]])

# Create a pickle per bird
# (* is replaced with _ when naming output files)
if not os.path.exists(_out_dir):
    os.makedirs(_out_dir)
for k, v in loaded_data.items():
        pkl_path = os.path.join(_out_dir, k.replace("*","_")+".pkl")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(v, pkl_file)

# Print a summary
print("Total labels: %d"%(len(loaded_data)))
print("Total items, at start: %d"%(total_loaded_items))
no_dupe = 0
for k,v in loaded_data.items():
    no_dupe += len(v)
print("Total items, at end: %d"%(no_dupe))

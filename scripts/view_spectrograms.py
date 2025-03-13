import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

# Function to load the list of 2D NumPy arrays from a pickle file
def load_arrays_from_pickle(filename):
    with open(filename, 'rb') as file:
        arrays = pickle.load(file)
    return arrays

# Function to display the array as a colormesh
def display_array_as_colormesh(arr):
    plt.pcolormesh(arr)
    plt.ylabel('Frequency bin')
    plt.xlabel('Sample')
    plt.title(f'Spectrogram {current_index + 1} of {len(arrays)}')
    plt.colorbar(label='Intensity')

#    plt.pcolormesh(arr, cmap='viridis')
#    plt.colorbar()
    #plt.title(f"Array {current_index + 1} of {len(arrays)}")
    plt.draw()

# Key event function to navigate through arrays
def on_key(event):
    global current_index

    if event.key == 'right':
        current_index = (current_index + 1) % len(arrays)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(arrays)

    plt.clf()  # Clear the current figure
    display_array_as_colormesh(arrays[current_index])
    plt.show()

# Initialize

# Check if there is exactly one command line argument (excluding the script name)
if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print("This script requires exactly one command line argument.")
    sys.exit()

#filename = '../dataset/LTT/B_NL.pkl'  # Replace with your pickle file
arrays = load_arrays_from_pickle(filename)

# Start with the first array
current_index = 0

# Plot the first array
plt.figure()
display_array_as_colormesh(arrays[current_index])

# Connect the key press event
cid = plt.gcf().canvas.mpl_connect('key_press_event', on_key)

# Display the plot
plt.show()

# Note: Replace 'your_pickle_file.pkl' with the actual path to your pickle file.

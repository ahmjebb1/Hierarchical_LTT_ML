import numpy as np
import pandas as pd
import tqdm
import pickle
import os

from scipy.io import wavfile
from scipy import signal

def get_spectrogram(filename):
    sample_rate, data = wavfile.read(filename)

    # Check if stereo or mono:
    if len(data.shape) == 2:
        # Convert stereo to mono by averaging channels:
        data = data.mean(axis=1) 

    # Generate the spectrogram:
    frequencies, times, spectrogram = signal.spectrogram(data, sample_rate)
    spectrogram = spectrogram[1:,]
    frequencies = frequencies[1:]

    return (frequencies, times, spectrogram)


df = pd.read_csv('great-tit-hits.csv')

count = 0
data = {}
bird_ids = set()

for index, row in tqdm(df.iterrows(), total=len(df)):
    fname = "WAV/"+row['filename'] + ".wav"
    bird_id = row['ID']
    bird_ids.add(bird_id)
    count += 1

for bird_id in tqdm(bird_ids):
    filtered = df[df['ID'] == bird_id]
    classes = set(filtered['class_id'])

    for class_id in classes:
        filtered_2 = filtered[filtered['class_id'] == class_id]
        spects_temp = []
        for index, row in tqdm(filtered_2.iterrows(), total=len(filtered_2)):
            fname = "WAV/"+row['filename'] + ".wav"
            bird_id_csv = row['ID']
            if bird_id != bird_id_csv: print("Error: unmatched bird_id")

            frequencies, times, spectrogram = get_spectrogram(fname)

            spects_temp.append(spectrogram)

        # Save the list of spectrograms for this bird:
        class_num = class_id.split('_')[1]
        try:
            os.mkdir(f"output/{class_num}")
        except OSError as error:
            pass
        out_file = f"output/{class_num}/{bird_id}.pkl"
        
        with open(out_file, 'wb') as f:
            pickle.dump(spects_temp, f)


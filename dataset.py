import os
import pickle
import random
import sys

import numpy as np

from torch.utils.data import Dataset, random_split

from util import log

class LTTMLSpectrogramDataset(Dataset):
    def __init__(self, config):
        self._config = config
        self.data_dir = self._config.data_dir
        self.classes = [ f[:-4] for f in os.listdir(self.data_dir) if not f==".DS_Store" ]
        log(0,"Number of classes in dataset: "+str(len(self.classes)))
        log(-1,"classes: "+str(self.classes))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        # self.pickle_files = self._get_pickle_files()
        self.pickle_files = [c+".pkl" for c in self.classes]
        log(-1,"pickle files: "+str(self.pickle_files))

        self.data_augmentation_amount =self._config.augmentation
        self.use_augmented = False

        self.unaugmented_data = []
        self.augmented_data = []

        self.train_data = []
        self.validation_data = []
        self.test_data = []

        self.untransformed_data = []
        self.data = []
        data_shape = [9,12,8,16,30,4,43,48,32,8,10,16,30,10,29,29,10,46,34,22,103,18,6,19,20,11,23,25,1,38,6,42,10,43,9,27,12,2,32]

        max_classes = len(data_shape)

        n = 0
        # Iterate over all the pickle files to pull out (spectrogram, label) pairs:
        for pickles in self.pickle_files:
            if pickles==".pkl":
                log(2, "a pickle file without a filename made it through")
                continue
            log(-1, "load pickle: "+str(pickles))
            with open(self.data_dir+"/"+pickles, "rb") as file:
                spectrograms = pickle.load(file)
                pairs = [
                            (spectrogram,
                            #  self.class_to_idx[os.path.basename(os.path.dirname(pickles))])
                            self.class_to_idx[pickles[:-4]])
                            for spectrogram in spectrograms
                        ]
            # Check if there are too many - truncate if soo
            if len(pairs) > self._config.max_examples_per_class:
                random.shuffle(pairs)
                pairs = pairs[:self._config.max_examples_per_class]
        #    if n < max_classes: 
        #        pairs = pairs[:data_shape[n]]
        #    else:
        #        break
            # Check if there are enough - add to the dataset if so
            if len(pairs) >= self._config.min_examples_per_class:
                self.unaugmented_data.extend(pairs)
            n = n + 1
        if self._config.max_classes > 0:
            self.truncate_unaugmented(self._config.max_classes)

        loaded_classes = set([ x[1] for x in self.unaugmented_data ])
        log(0, f"Classes in use: {len(loaded_classes)}")
            
        if self._config.min_classes > 0:
            if len(loaded_classes) < self._config.min_classes:
                log(2, f"Too few classes to proceed! Minimum: {self._config.min_classes}")
                sys.exit(1)

        # Set the transform (it's not being used now)
        self.transform = None

        # Split the unaugmented data. This is done here so the dataset can tell the
        # application which data are OK to train on and which should be reserved for
        # validation/test, rather than the application splitting the whole dataset itself.
        #
        # This is because I can't validate on augmented data because it likely contains
        # very similar examples to the training set, so the augmentation has to happen
        # *after* the split.
        train_size = int(0.7 * len(self.unaugmented_data))
        val_size = int(0.15 * len(self.unaugmented_data))
        test_size = len(self.unaugmented_data) - train_size - val_size

        tr, v, te = random_split(self.unaugmented_data, [train_size, val_size, test_size])

        # tr, v, te are Pytorch "Subset"s - so I have to rebuild the lists from them:
        self.train_data = [x for x in tr]
        self.validation_data = [x for x in v]
        self.test_data = [x for x in te]

        # Now I can augment the training set:
        for d in self.train_data:
            augments = self.data_augmentation(d, self.data_augmentation_amount)
            self.augmented_data.extend(augments)

        # And add the augmented (spectrogram,label) pairs to the training set:
        self.train_data.extend(self.augmented_data)

        # Build the final data in sequence [train, val, test]:
        self.untransformed_data.extend(self.train_data)
        self.untransformed_data.extend(self.validation_data)
        self.untransformed_data.extend(self.test_data)

        # Apply the transformation (transpose, pad and normalise) to get the
        # actual, final final dataset:
        self.data = [(self.data_transformation(s),l) for (s,l) in self.untransformed_data]

        log(-1, f"first data item data: "+str(len(self.data[0][0])))
        log(-1, f"first data item label: "+str(self.data[0][1]))
        self.print_stats()

    def _get_pickle_files(self):
        pickle_files = []
        for cls in self.classes:
            class_dir = os.path.join(self.data_dir, cls)
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.pkl')]
            pickle_files.extend(files)
        return pickle_files

    def get_training_indices(self):
        start = 0
        end = len(self.train_data)
        return list(range(start, end))

    def get_validation_indices(self):
        start = len(self.train_data)
        end = start + len(self.validation_data)
        return list(range(start, end))

    def get_test_indices(self):
        start = len(self.train_data) + len(self.validation_data)
        end = start + len(self.test_data)
        return list(range(start, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx][0]
        if self._config.model_type=="transformer":
            item=item.squeeze(0)
        label = self.data[idx][1]

        # Apply transformations if any
        # This moved to __init__ now
        if self.transform:
            spectrogram = self.transform(item)

        return item, label

    def truncate_unaugmented(self, num_classes):
        classes = set()
        for (_, l) in self.unaugmented_data:
            classes.add(l)
        subset = random.sample(sorted(classes), num_classes)
        new_data = [ (s, l) for (s, l) in self.unaugmented_data if l in subset ]
        self.unaugmented_data = new_data

    def print_stats(self):
        unaugmented_stats = {}
        augmented_stats = {}

        for (s, l) in self.unaugmented_data:
            if l in unaugmented_stats:
                unaugmented_stats[l] += 1
            else:
                unaugmented_stats[l] = 1

        for (s, l) in self.augmented_data:
            if l in augmented_stats:
                augmented_stats[l] += 1
            else:
                augmented_stats[l] = 1

        augmented_stats = {key: augmented_stats[key] for key in sorted(augmented_stats)}
        total = 0
        log(-1,"Raw data\n---------")
        for value, count in unaugmented_stats.items():
            log(-1,f"[{self.classes[value]}]\t{value}: {count}")
            total = total + count
        log(-1,f"total: {total}")

        total = 0
        log(-1,"Augmented data\n--------------")
        for value, count in augmented_stats.items():
            log(-1,f"[{self.classes[value]}]\t{value}: {count}")
            total = total + count
        log(-1,f"total: {total}")

# Takes an untransformed spectrogram and returns a list of num_augments modified
# spectrograms. Currently it's a random masking of parts of the spectrogram in both the
# time and frequency axes. This should happen before the transformation step as it
# expects frequency axis 0 and time axis 1 (the transpose operation happens in transformation step).
    def data_augmentation(self, pair, num_augments):

        time_masks = 2
        freq_masks = 2
        augments = []

        spectrogram, label = pair

        for _ in range(num_augments):
            augmented_spectrogram = np.copy(spectrogram)
            # Apply time masking
            for _ in range(time_masks):
                t = np.random.randint(0, spectrogram.shape[1])
                t0 = max(0, t - 10)
                t1 = min(spectrogram.shape[1], t + 10)
                augmented_spectrogram[:, t0:t1] = 0

            # Apply frequency masking
            for _ in range(freq_masks):
                f = np.random.randint(0, spectrogram.shape[0])
                f0 = max(0, f - 5)
                f1 = min(spectrogram.shape[0], f + 5)
                augmented_spectrogram[f0:f1, :] = 0

            augments = augments + [(augmented_spectrogram,label)]

        return augments

    # These magic numbers come from running the script without the normalisation below, and with the
    # call to get_norm_stats() in birdcall_train.py uncommented.
    def data_transformation(self, numpy_array):

        norm_mean=self._config.normalization_mean # 0.24706475
        norm_std=self._config.normalization_std   # 1.6275752

        # Pad the array in axis 0 (time) with 0s to be the same shape (same number of samples) as the
        # HuggingFace trained model, and normalise as per Yuan Gong

        # Desired shape after padding
        desired_shape = (self._config.spectrogram_samples, self._config.spectrogram_bins)

        # truncate larger spectrograms to maximum of desired_shape:
        original_array = numpy_array.transpose()[:desired_shape[0],:desired_shape[1]]

        # pad smaller spectrograms to desired_shape:
        axis0_padding_amount = desired_shape[0] - original_array.shape[0]
        axis1_padding_amount = desired_shape[1] - original_array.shape[1]

#        log(-1, f"Padding axes in data point by [{axis0_padding_amount},{axis1_padding_amount}]")

        axis0_pre_pad = random.randint(0, axis0_padding_amount)
        axis0_post_pad = axis0_padding_amount - axis0_pre_pad
        # Calculate the amount of padding needed for each dimension. Should not need to pad frequency axis (1)
        pad_width = [(0, max(0, axis0_padding_amount)),  # Padding along the first axis
                    (0, max(0, axis1_padding_amount))]  # Padding along the second axis

        # Pad the array
        padded_array = np.pad(original_array, pad_width, mode='constant')

        # From https://github.com/YuanGongND/ast/blob/master/src/dataloader.py:
        # (Comment out to regenerate norm stats for the magic numbers above)
        if self._config.normalize:
            padded_array = (padded_array - norm_mean) / (norm_std * 2)

        return np.expand_dims(padded_array, axis=0) # add the channel dimension

    def get_norm_stats(self):
        mean=[]
        std=[]
        for i, (audio_input, labels) in enumerate(self.data):
            cur_mean = np.mean(audio_input)
            cur_std = np.std(audio_input)
            mean.append(cur_mean)
            std.append(cur_std)
            #print(cur_mean, cur_std)
        log(0,f"normalization_mean: {np.mean(mean)}")
        log(0,f"normalization_std: {np.mean(std)}")

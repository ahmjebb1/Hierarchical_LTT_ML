# ltt-ml
Long-tailed Tit Call Machine Learning Project

# data
Raw LTT data is processed using the updated [Lotti App](https://github.com/bubblecontrol/lotti_app). Raw GT data from the [Wytham Wood dataset](https://nilomr.github.io/great-tit-hits/), hosted at OSF: https://osf.io/n8ac9/files/osfstorage. This data downloads as four partial zips, you have to concatenate these together and then "fix" them if you are using standard tools. E.g.:
```
cat song_files.zip.part1 song_files.zip.part2 song-files.zip.part3 song-files.zip.part4 >> song-files.broken.zip
zip -FF song-files.broken.zip --out song-files.zip
cd $DESTINATION
unzip $SOURCE/song-files.zip
```

# tool scripts
- `scripts/prep_wytham_data.py` - script to convert the raw GT data to the pickle format used by the data loader (the same as LTTs)
- `scripts/datasort.py` - use this to read through all of the raw selections/images files produced by the Lotti app and build pickles containing the calls, one pickle per bird (class).
- `scripts/get_LTT_class_sizes.py` - point this to the LTT files produced above - it will print out a list with the length of each class
- `scripts/reduce_GTc_to_GTr.py` - edit this so it's list of class sizes is accurate (paste the output of above in the right place) and it will produce a great tit dataset the same shape as the LTT dataset


# ml scripts
The following scripts can be run:
* `train.py` - train a model. You need to provide a configuration YAML file.
* `infer.py` - TBD - use an existing model to classify a datum. You need to provide a configuration file, a saved model, and an input pickle with one or more spectrograms in a list.

# dataset
A collection of [pickle files](https://docs.python.org/3/library/pickle.html) (.pkl) comprises the preprocessed audio data, repackaged after processing in the normal way with Raven Lite and Lotti, or after using the `prep_wytham_data.py` script on the Wytham Great Tit dataset. Each file, e.g. `B_GL.pkl`, is a pickle of a list. The list contains numpy arrays, typically of shape (128,300), representing the spectrogram of a call (128 frequency bins and 300 samples). Each list is all the calls for a single bird.

**Important: before using a new or modified dataset, normalisation values should be calculated - see "calculating normalisation" below.**

# configurations
Configuration is handled by putting settings in a [YAML](https://en.wikipedia.org/wiki/YAML) file.

<details>

<summary>Example configuration file</summary>

```yaml
# Hardware
use_mps: True   # Metal shader, Macintosh
use_cuda: True  # CUDA - preferred

# Dataset parameters
data_directory: "dataset"
spectrogram_samples: 300 # 1024 -transformer, 300 - cnn
spectrogram_bins: 128
min_examples_per_class: 0
max_examples_per_class: 1000
augmentation: 0
normalization_mean: 0.7941427826881409
normalization_std: 2.8298227787017822
normalize: True
calculate_norm_stats: False
max_classes: 0          # truncate the number of classes in the dataset to this

# Model parameters
model: "cnn"
num_outputs: 40 # 527 if using ast, 40 for CNN
conv1_size: 16
conv2_size: 32
hidden: 128
use_softmax: False # False for transformer

# Training parameters
batch_size: 1
learning_rate: 0.000005
epochs: 16
dropout_rate: 0.3

# Other parameters
debug: False # not implemented
mlflow_user: "admin"
mlflow_pw: "admin1"
mlflow_uri: "http://127.0.0.1:5000"
mlflow_expname: "birdcall_2"
mlflow_description: "example configuration"
```

</details>

<details>

<summary>Configuration options with descriptions</summary>

 | field | example Value | description |
 |-------|:-------------:|-------------|
 | use_mps | True | Whether to try to use the Metal Performance Shader to do GPU acceleration on Macintosh systems | 
 | use_cuda | True | Whether to try to use Compute Uniform Device Architecture on Nvidia-based systems | 
 | data_directory | dataset | Where the pickle files are located | 
 | spectrogram_samples | 300 | Size of spectrogram - smaller data are padded | 
 | spectrogram_bins | 128 | Size of spectrogram - smaller data are padded | 
 | min_examples_per_class | 0 | Discard classes (birds) with fewer than min_examples_per_class | 
 | max_examples_per_class | 1000 | Limit data per class (bird) to max_examples_per_class | 
 | augmentation | 0 | n-fold augmentation (make n copies of each datum) | 
 | normalization_mean | 0.7941427826881409 | Pre-calcuated normalisation mean value | 
 | normalization_std | 2.8298227787017822 | Pre-calculated normalisation stddev value | 
 | normalize | True | If true, apply normalisation to each datum (normally should be used) | 
 | calculate_norm_stats | False | If true, recalculate and report the norm stats (normally should be off) | 
 | max_classes | 0 | Limit the number of classes to max_classes | 
 | model | cnn | Type of model - 'cnn', 'transformer', or 'svm' at the moment | 
 | num_outputs | 40 | Number of outputs - should be >= number of classes (birds) | 
 | conv1_size | 16 | CNN - size of first convolutional layer | 
 | conv2_size | 32 | CNN - size of second convolutional layer | 
 | hidden | 128 | CNN - size of hidden layer | 
 | use_softmax | False | Apply softmax to outputs to get probabilites | 
 | batch_size | 1 | Number of data to include in a batch | 
 | learning_rate | 5e-06 | Learning rate | 
 | epochs | 16 | Number of learning epochs | 
 | dropout_rate | 0.3 | Dropout (regularisation) rate | 
 | debug | False | Display debug info (not used - it's a flag in the code at the moment) | 
 | mlflow_user | admin | Username for MLFlow | 
 | mlflow_pw | admin1 | Password for MLFlow | 
 | mlflow_uri | http://127.0.0.1:5000 | API endpoint for MLFlow | 
 | mlflow_expname | birdcall_2 | Experiment name in MLFlow (for grouping runs) | 
 | mlflow_description | example configuration | Description of this run to store in MLFlow | 

</details>


# calculating normalisation

To calculate the normalisation, set "normalize" to False and "calculate_norm_stats" to True in your configuration file and run the training script. The script will report the stats and exit - you can then use these updated stats in your configuration files for this dataset.

<details>
<summary>Example normalisation</summary>

```
python train.py test_config.yaml

Info:	MPS device available.
Info:	Device: mps
Info:	Loaded data. Total size: 893 (Applied 0x additional augmentation)
Info:	Training examples: 625 Validation set: 133 Test set: 135
Info:	calculating normalization stats.
Info:	normalization_mean: 0.7941428422927856
Info:	normalization_std: 2.8298227787017822
```
</details>

# MLFlow integration
The script interacts with the MLFlow ML ops front end. MLFlow collects information about each training run and presents them for comparison,
facilitating the search for good models and parameters to solve a particular problem. The MLFlow server is a separate piece of software that
runs like a web server. You can run this locally or use a remote server. To run locally, you start the server using
`mlflow server --host 127.0.0.1` and then use a web browser to navigate to [`http://localhost:5000`](http://localhost:5000). The training script
looks at the YAML config to know where to send the information to:

|setting | example value | description|
|--------|---------------|------------|
|mlflow_user|```"admin"```|User name|
|mlflow_pw|```"admin1"```|user password|
|mlflow_uri|```"http://127.0.0.1:5000"```|Endpoint for MLFlow|
|mlflow_expname|```"birdcall_2"```|Experiment name to add the run to in MLFlow|
|mlflow_description|```"example configuration"```|A description that is visible in the experiment|

![Screenshot 2024-06-17 at 13 37 01](https://github.com/RSE-Sheffield/cmi-birdcall/assets/147408699/6cafba9a-c39d-4847-a4a6-c93fcbe03dd0)

# inference

TBD

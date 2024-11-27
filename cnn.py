import torch.nn as nn
import torch.nn.functional as F

class BirdCNNModel(nn.Module):
    def __init__(self, config):
        self.conv1_out = config.conv1_size #4 #16
        self.conv2_out = config.conv2_size #8 #32
        self.hidden = config.hidden #128
        self.dropout_rate = config.dropout_rate #0.67
        self.spectrogram_samples = config.spectrogram_samples
        self.spectrogram_bins = config.spectrogram_bins
        self.fc1_bins_in = self.spectrogram_bins // 4
        self.fc1_sams_in = self.spectrogram_samples // 4
        self.num_outputs = config.num_outputs

        super(BirdCNNModel, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv1_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv1_out, out_channels=self.conv2_out, kernel_size=3, stride=1, padding=1)
        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # fully connected layers n x m original spectrogram, pooled twice so 1/4ed dims (n/4*m/4).
        # 32 out channels from conv2, so 32 * n/4 * m/4 inputs to FC1
        self.fc1 = nn.Linear(self.fc1_bins_in * self.fc1_sams_in * self.conv2_out, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.num_outputs)  #  N classes
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):        
        # convolve input, apply relu activiation and pool:
        x = self.pool(F.relu(self.conv1(x)))
        # convolve previous layer, apply relu activiation and pool:
        x = self.pool(F.relu(self.conv2(x)))
        # flatten input for fully connected layers
        x = x.view(-1, self.fc1_bins_in * self.fc1_sams_in * self.conv2_out) 
        # pass data through fc1 with relu activation
        x = F.relu(self.fc1(x))
        # pass through 2nd layer to get the N outputs
        x = self.fc2(x)
        # return the outputs ...
        return x

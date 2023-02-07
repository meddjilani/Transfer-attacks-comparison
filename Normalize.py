import torch.nn as nn

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, input):
        channels = input.size(1)
        input_norm = input.clone()
        for channel in range(channels):
            input_norm[:,channel] = (input[:,channel] - self.mean[channel]) / self.std[channel] 

        return input_norm
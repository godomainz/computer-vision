# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                       0.5)), ])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root='./data', download=True,
                       transform=transform)  # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True,
                                         num_workers=2)  # We use dataLoader to get the images of the training set batch by batch.


# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Defining the Generator

class G(nn.Module):
    feature_maps = 512
    kernel_size = 4
    bias = False

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, self.feature_maps, self.kernel_size, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps, self.feature_maps / 2, self.kernel_size, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps / 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps / 2, (self.feature_maps / 2) / 2, self.kernel_size, 2, 1,
                               bias=self.bias),
            nn.BatchNorm2d((self.feature_maps / 2) / 2), nn.ReLU(True),
            nn.ConvTranspose2d((self.feature_maps / 2) / 2, ((self.feature_maps / 2) / 2) / 2, self.kernel_size, 2, 1,
                               bias=self.bias),
            nn.BatchNorm2d((self.feature_maps / 2) / 2) / 2, nn.ReLU(True),
            nn.ConvTranspose2d(((self.feature_maps / 2) / 2) / 2, 3, self.kernel_size, 2, 1, bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Creating the generator
netG = G()
netG.apply(weights_init)
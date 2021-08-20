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


input_vector = 100


# Defining the Generator
class G(nn.Module):
    feature_maps = 512
    kernel_size = 4
    stride = 2
    padding = 1
    bias = False

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_vector, self.feature_maps, self.kernel_size, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps, self.feature_maps / 2, self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(self.feature_maps / 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps / 2, (self.feature_maps / 2) / 2, self.kernel_size, self.stride,
                               self.padding,
                               bias=self.bias),
            nn.BatchNorm2d((self.feature_maps / 2) / 2), nn.ReLU(True),
            nn.ConvTranspose2d((self.feature_maps / 2) / 2, ((self.feature_maps / 2) / 2) / 2, self.kernel_size,
                               self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d((self.feature_maps / 2) / 2) / 2, nn.ReLU(True),
            nn.ConvTranspose2d(((self.feature_maps / 2) / 2) / 2, 3, self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Creating the generator
netG = G()
netG.apply(weights_init)


class D(nn.Module):
    feature_maps = 64
    kernel_size = 4
    stride = 2
    padding = 1
    bias = False
    inplace = True

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, self.feature_maps, self.kernel_size, self.stride, self.padding, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps, self.feature_maps * 2, self.kernel_size, self.stride, self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * 2), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * (2 * 2), self.kernel_size, self.stride, self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * (2 * 2)), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * (2 * 2), self.feature_maps * (2 * 2 * 2), self.kernel_size, self.stride,
                      self.padding, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * (2 * 2 * 2)), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * (2 * 2 * 2), 1, self.kernel_size, 1, 0, bias=self.bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


# Creating the discriminator
netD = D()
netD.apply(weights_init)

# Training the DCGANs
criterian = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.002, betas=(0.5, 0.999))

nb_epochs = 25

for epoch in range(nb_epochs):
    for i, data in enumerate(dataloader, 0):
        # 1st Step: Updating the weights of the neural network of the discriminator
        netD.zero_grad()

        # Training the discriminator with a real images of the dataset
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterian(output, target)

        # Training the discriminator with a fake images generated by the generator
        noise = Variable(torch.randn(input.size()[0], input_vector, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netG(fake.detach())
        errD_fake = criterian(output, target)

        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # 2nd Step: Updating the weights of the neural network of the generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterian(output, target)
        errG.backward()
        optimizerG.step()
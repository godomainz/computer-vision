# Importing the libraries
from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from generator import G
from discriminator import D
import os
from PIL import Image

batchSize = 64  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).
input_vector = 100
nb_epochs = 500
# Creating the transformations
transform = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5,
                                                                            0.5)), ])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.


def pil_loader_rgba(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


# Loading the dataset
dataset = dset.ImageFolder(root='./data', transform=transform, loader=pil_loader_rgba)
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


def is_cuda_available():
    return torch.cuda.is_available()


def is_gpu_available():
    if is_cuda_available():
        if int(torch.cuda.device_count()) > 0:
            return True
        return False
    return False


def num_of_gpus_available():
    if is_gpu_available():
        return int(torch.cuda.device_count())
    return 0


device = torch.device("cuda:0" if is_gpu_available() else "cpu")
ngpu = num_of_gpus_available()
print("Number of GPUs available: " + str(ngpu))

# Create results directory
def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


# Creating the generator
netG = G(input_vector, ngpu).to(device)
netG.apply(weights_init)

# Creating the discriminator
netD = D(ngpu).to(device)
netD.apply(weights_init)

if is_gpu_available():
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Training the DCGANs

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

generator_model = 'generator_model'
discriminator_model = 'discriminator_model'


def save_model(epoch, model, optimizer, error, filepath, noise=None):
    if os.path.exists(filepath):
        os.remove(filepath)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': error,
        'noise': noise
    }, filepath)


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        return torch.load(filepath)
    return None


def main():
    print("Device name : " + torch.cuda.get_device_name(0))
    for epoch in range(nb_epochs):

        for i, data in enumerate(dataloader, 0):
            checkpointG = load_checkpoint(generator_model)
            checkpointD = load_checkpoint(discriminator_model)
            if checkpointG:
                netG.load_state_dict(checkpointG['model_state_dict'])
                optimizerG.load_state_dict(checkpointG['optimizer_state_dict'])
            if checkpointD:
                netD.load_state_dict(checkpointD['model_state_dict'])
                optimizerD.load_state_dict(checkpointD['optimizer_state_dict'])

            # 1st Step: Updating the weights of the neural network of the discriminator

            netD.zero_grad()

            # Training the discriminator with a real image of the dataset
            real, _ = data
            if is_gpu_available():
                input = Variable(real.cuda()).cuda()
                target = Variable(torch.ones(input.size()[0]).cuda()).cuda()
            else:
                input = Variable(real)
                target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)

            # Training the discriminator with a fake image generated by the generator
            if is_gpu_available():
                noise = Variable(torch.randn(input.size()[0], input_vector, 1, 1)).cuda()
                target = Variable(torch.zeros(input.size()[0])).cuda()
            else:
                noise = Variable(torch.randn(input.size()[0], input_vector, 1, 1))
                target = Variable(torch.zeros(input.size()[0]))
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            # Backpropagating the total error
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # 2nd Step: Updating the weights of the neural network of the generator
            netG.zero_grad()
            if is_gpu_available():
                target = Variable(torch.ones(input.size()[0])).cuda()
            else:
                target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                epoch, nb_epochs, i, len(dataloader), errD.data, errG.data))
            save_model(epoch, netG, optimizerG, errG, generator_model, noise)
            save_model(epoch, netD, optimizerD, errD, discriminator_model, noise)

            if i % 100 == 0:
                create_dir('results')
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)


if __name__ == "__main__":
    main()

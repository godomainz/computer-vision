import torch
from torch.autograd import Variable
from generator import G
from gan import is_gpu_available, num_of_gpus_available, device, generator_model, load_checkpoint
import torch.nn as nn
import torchvision.utils as vutils

input_vector = 100
device = device
noise = Variable(torch.randn(1, input_vector, 1, 1, device=device))
ngpu = num_of_gpus_available()

# checkpointG = load_checkpoint(generator_model)
netG = G(input_vector, ngpu).to(device)
# if is_gpu_available():
#     netG = nn.DataParallel(netG, list(range(ngpu)))
# if checkpointG:
#             netG.load_state_dict(torch.load(generator_model))
netG.load_state_dict(torch.load(generator_model))

fake = netG(noise)
vutils.save_image(fake.data, 'deepfake.png', normalize=True)
import torch.nn as nn
class D(nn.Module):
    feature_maps = 64
    kernel_size = 4
    stride = 2
    padding = 1
    bias = True
    inplace = True

    def __init__(self, ngpu=0):
        super(D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(4, self.feature_maps, self.kernel_size, self.stride, self.padding, bias=self.bias),
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
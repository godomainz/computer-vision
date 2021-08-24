import torch.nn as nn
class G(nn.Module):
    feature_maps = 512
    kernel_size = 4
    stride = 2
    padding = 1
    bias = False

    def __init__(self, input_vector):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_vector, self.feature_maps, self.kernel_size, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps, int(self.feature_maps // 2), self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(int(self.feature_maps // 2)), nn.ReLU(True),
            nn.ConvTranspose2d(int(self.feature_maps // 2), int((self.feature_maps // 2) // 2), self.kernel_size, self.stride,
                               self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(int((self.feature_maps // 2) // 2)), nn.ReLU(True),
            nn.ConvTranspose2d((int((self.feature_maps // 2) // 2)), int(((self.feature_maps // 2) // 2) // 2), self.kernel_size,
                               self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(int((self.feature_maps // 2) // 2) // 2), nn.ReLU(True),
            nn.ConvTranspose2d(int(((self.feature_maps // 2) // 2) // 2), 4, self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
import torch
import torch.nn as nn
import torchvision
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained = True)
        self.vgg_list = list(list(self.vgg.children())[0].children())
        self.vgg = nn.Sequential(*self.vgg_list)

        self.conv1x1 = conv_unit(in_ch = 512, out_ch = 512, kernel_size = 1, activation = 'relu')
    
    def forward(self, x):
        # input = (3, 256, 448)
        x = self.vgg(x)
        # (512, 8, 14)
        x = self.conv1x1(x)
        # (same)
        return x

# encoder = Encoder().to('cuda:0')
# summary(encoder, (3, 256, 448))

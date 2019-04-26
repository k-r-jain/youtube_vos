import torch
import torch.nn as nn
import torchvision
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class Initializer(nn.Module):
    def __init__(self):
        super(Initializer, self).__init__()

        self.to_three_channels = conv_unit(in_ch = 4, out_ch = 3, kernel_size = 3, activation = 'relu', padding = 1)
        self.vgg = torchvision.models.vgg16(pretrained = True)
        self.vgg_list = list(list(self.vgg.children())[0].children())
        self.vgg = nn.Sequential(*self.vgg_list)

        self.cell = conv_unit(in_ch = 512, out_ch = 512, kernel_size = 1, activation = 'relu')
        self.hidden = conv_unit(in_ch = 512, out_ch = 512, kernel_size = 1, activation = 'relu')
    
    def forward(self, x):
        # input = (4, 256, 448)
        x = self.to_three_channels(x)
        x = self.vgg(x)
        # (512, 8, 14)
        c = self.cell(x)
        # (same as x)
        h = self.hidden(x)
        # (same as x)
        
        return c, h

# initializer = Initializer().to('cuda:0')
# summary(initializer, (3, 256, 448))

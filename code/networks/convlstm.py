import torch
import torch.nn as nn
import torchvision
import numpy as np
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class ConvLSTMCell(nn.Module):
    def __init__(self, channels = 512, height = 8, width = 14, device = 'cuda:0'):
        super(ConvLSTMCell, self).__init__()

        # Convolutions for gate computations
        self.Wxi = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Whi = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Wxf = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Whf = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Wxc = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Whc = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Wxo = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)
        self.Who = conv_unit(in_ch = channels, out_ch = channels, kernel_size = 3, padding = 1, activation = None, batch_norm = False).to(device)

        # Matrices used for Hadamard product used in gate computations
        self.Wci = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wcf = torch.randn((channels, height, width), requires_grad = True).to(device)
        self.Wco = torch.randn((channels, height, width), requires_grad = True).to(device)
        nn.init.kaiming_uniform_(self.Wci)
        nn.init.kaiming_uniform_(self.Wcf)
        nn.init.kaiming_uniform_(self.Wco)

        # Since paper uses ReLU instead of the standard TanH function
        self.gate_activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c_prev, h_prev):
        # x = tensor[:, :512, :, :].to('cuda:0')
        # c_prev = tensor[:, 512:1024, :, :].to('cuda:0')
        # h_prev = tensor[:, 1024:, :, :].to('cuda:0')
        # print(x.size())
        # print(self.Wxi.device)
        # c_prev = torch.randn((2, 512, 8, 14)).to('cuda:0')
        # h_prev = torch.randn((2, 512, 8, 14)).to('cuda:0')

        i = self.Wxi(x) + self.Whi(h_prev) + (self.Wci * c_prev)
        i = self.sigmoid(i)

        f = self.Wxf(x) + self.Whf(h_prev) + (self.Wcf * c_prev)
        f = self.sigmoid(f)

        c = self.gate_activation(self.Wxc(x) + self.Whc(h_prev))
        c = (f * c_prev) + (i * c)

        o = self.Wxo(x) + self.Who(h_prev) + (self.Wco * c)
        o = self.sigmoid(o)

        h = o * self.gate_activation(c)
        return c, h


# convlstmcell = ConvLSTMCell().to('cuda:0')
# summary(convlstmcell, (512*3, 8, 14))

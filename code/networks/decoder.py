import torch
import torch.nn as nn
import torchvision
import numpy as np
# from torchsummary import summary
if __name__ == '__main__':
    from nnutils import conv_unit
else:
    from .nnutils import conv_unit

class Decoder(nn.Module):
	def __init__(self, input_channels = 512, input_res = (8, 14), init_channels = 512, shrink_per_block = 2, output_channels = 1, output_res = (256, 448)):
		super(Decoder, self).__init__()
		self.input_channels = input_channels
		self.input_res = np.array(input_res)
		self.init_channels = init_channels
		self.shrink_per_block = shrink_per_block
		self.output_channels = output_channels
		self.output_res = np.array(output_res)
		self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
		self.net = self.network(self.input_channels, self.input_res, self.init_channels, self.shrink_per_block, self.output_channels, self.output_res)

	def network(self, input_channels, input_res, init_channels, shrink_per_block, output_channels, output_res):
		modules = []
		prev_channels = input_channels
		# print('0', prev_channels, input_channels, input_res, output_res)
		while True:
			# print('1', prev_channels, init_channels)
			
			modules.append(conv_unit(in_ch = prev_channels, out_ch = init_channels, kernel_size = 5, stride = 1, padding = 2))

			# print('2', prev_channels, init_channels)

			if np.array_equal(input_res, output_res):
				modules.append(conv_unit(in_ch = init_channels, out_ch = output_channels, kernel_size = 5, stride = 1, padding = 2, activation = 'sigmoid'))
				# print('3', prev_channels, init_channels)

				break
			else:
				modules.append(conv_unit(in_ch = init_channels, out_ch = init_channels, kernel_size = 5, stride = 1, padding = 2))
				# print('4', prev_channels, init_channels)

				modules.append(self.upsample)
				input_res *= 2
			
			prev_channels = init_channels
			if init_channels > 64:
				init_channels = int(init_channels / shrink_per_block)
				# print('5', prev_channels, init_channels)


		return nn.Sequential(*modules)

	def forward(self, h):
		return self.net(h)

# decoder = Decoder().to('cuda:0')
# summary(decoder, (512, 8, 14))
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from vos import YouTubeVOSLoader
from networks.initializer import Initializer
from networks.encoder import Encoder
from networks.convlstm import ConvLSTMCell
from networks.decoder import Decoder
import copy

import sys


machine = 'home'
# machine = 'newton'

if machine == 'home':
	root_data_dir = '/home/kartik/data/youtube_vos'
	save_models_dir = '/home/kartik/Desktop/vos_models_updated'
	save_segmented_video_dir = '/home/kartik/Desktop/vos_results_updated'
	loss_plot_save_path =  '/home/kartik/Desktop/loss_plot_updated.png'
	batch_size = 2
	num_workers = 8
	num_epochs = 1
	fraction = 0.01
	SAVE_EVERY_N_EPOCH = 20
	NUM_FRAMES = 5
	resize_dim = (128, 224)
	# resize_dim = (256, 448)

elif machine == 'newton':
	root_data_dir = '/home/course.cap6412/youtubeVOS'
	save_models_dir = '/home/cap6412.student4/pa2/vos_models_updated'
	save_segmented_video_dir = '/home/cap6412.student4/pa2/vos_results_updated'
	loss_plot_save_path =  '/home/cap6412.student4/pa2/loss_plot_updated.png'
	batch_size = 4
	num_workers = 4
	num_epochs = 100
	fraction = 0.50
	SAVE_EVERY_N_EPOCH = 20
	NUM_FRAMES = 5
	resize_dim = (256, 448)
	if len(sys.argv) > 1:
		VRAM = int(sys.argv[1])
		if VRAM < 10000:
			batch_size = 2
		elif VRAM < 20000:
			batch_size = 4
		elif VRAM > 30000:
			batch_size = 16
		print('VRAM', VRAM, 'batch size', batch_size)


save_models_name = 'vos_updated_'

mode = 'train'
device = 'cuda:0'
lr = 1e-4
shuffle_data = True

assert NUM_FRAMES >= 2



mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
image_transformation = transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])

# mean = (0.5, 0.5, 0.5)
# std = (0.5, 0.5, 0.5)
# mean = (0.0, 0.0, 0.0)
# std = (1.0, 1.0, 1.0)
# mask_transformation = transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
mask_transformation = transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor()])

ytvos = YouTubeVOSLoader(root = root_data_dir, mode = mode, fraction = fraction, image_transformation = image_transformation, mask_transformation = mask_transformation, num_frames = NUM_FRAMES)
ytvos = DataLoader(ytvos, batch_size = batch_size, shuffle = shuffle_data, num_workers = num_workers)
num_batches = len(ytvos)


###### MODEL OBJECTS ######

encoded_h = int(resize_dim[0] / 32)
encoded_w = int(resize_dim[1] / 32)

initializer = Initializer().to(device)
encoder = Encoder().to(device)
convlstmcell_encoder = ConvLSTMCell(height = encoded_h, width = encoded_w).to(device)
convlstmcell_decoder = ConvLSTMCell(channels = 512 * 2, height = encoded_h, width = encoded_w).to(device)
decoder = Decoder(input_channels = 512 * 2, input_res = (encoded_h, encoded_w), output_res = resize_dim).to(device)


cost_fn = nn.BCELoss()
optimizer = torch.optim.Adam(list(initializer.parameters()) + list(encoder.parameters()) + list(convlstmcell_encoder.parameters()) + list(convlstmcell_decoder.parameters()) + list(decoder.parameters()), lr = lr)


###########################

iter_count = -1
try:
	os.makedirs(save_models_dir, exist_ok = True)
except:
	pass

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


loss_plot = []
for epoch in range(num_epochs):
	print('Epoch', epoch, '-' * 30)
	epoch_avg_loss = 0.0
	for i, sample in enumerate(ytvos):

		# pass
		optimizer.zero_grad()
		x = sample['x'].to(device)
		y = sample['y'].to(device)
		# print(y, y.max(0)[1].size(), y.min(0)[1].size())
		t = sample['t'].to(device)
		name = sample['name']
		image_names = sample['image_names']
		# print(name, image_names)
		# print(x.size(), y.size(), t, name)

		# yhat_list, loss_list, loss_per_video = VOSBaseArch(x, y, t, initializer, encoder, convlstmcell, decoder, cost_fn)
		# print(len(yhat_list))
		# print(yhat_list[0].size())
		
		# loss_per_video.backward()
		# print('\r', sum(loss_list), end = '')


		yhat_list = [y[:, 0, :, :, :].squeeze(1).cpu().detach().numpy()]
		y_list = []
		loss_list = []
		loss_per_video = 0.0
		CURRENT_BATCH_SIZE = x.size(0)
		# print(x[:, 0, :, :, :].size(), y[:, 0, :, :, :].size())

		# All the frames in video
		ci, hi = torch.zeros((CURRENT_BATCH_SIZE, 512, encoded_h, encoded_w)).to(device), torch.zeros((CURRENT_BATCH_SIZE, 512, encoded_h, encoded_w)).to(device)
		for frame_id in range(x.size(1)):
			xi = x[:, frame_id, :, :, :]

			xi = encoder(xi)
			ci, hi = convlstmcell_encoder(xi, ci, hi)
		
		h_video = hi
		
		ci, hi = initializer(torch.cat((x[:, 0, :, :, :], y[:, 0, :, :, :]), dim = 1))
		ci = torch.cat((ci, h_video), dim = 1)
		hi = torch.cat((hi, h_video), dim = 1)

		for frame_id in range(1, x.size(1)):
			xi = x[:, frame_id, :, :, :]
			yi = y[:, frame_id, :, :, :]

			xi = encoder(xi)
			xi = torch.cat((xi, h_video), dim = 1)
			ci, hi = convlstmcell_decoder(xi, ci, hi)
			yhati = decoder(hi)

			# print(yhati.size())
			# print('unique', torch.unique(yhati), torch.unique(yi))
			loss = cost_fn(yhati, yi)
			loss_per_video += loss
			loss_list.append(loss.item())

			# yhati_copy = yhati
			# yhati_copy[yhati_copy >= 0.5] = 1.0
			# yhati_copy[yhati_copy < 0.5] = 0.0
			yhat_list.append(yhati.squeeze(1).cpu().detach().numpy())
			y_list.append(yi.squeeze(1).cpu().detach().numpy())


		loss_per_video.backward()
		optimizer.step()

		# ----------------- Saving annotations ---------------------------
		# print(len(yhat_list), yhat_list[0].shape)
		for i in range(yhat_list[0].shape[0]):
			# print('i', i)
			current_video_name = name[i]
			abs_video_path = os.path.join(save_segmented_video_dir, current_video_name)
			try:
				os.makedirs(abs_video_path, exist_ok = True)
			except:
				pass

			for j in range(len(yhat_list)):
				# print('j', j)

				gen_mask = yhat_list[j][i, :, :]
				# plt.imshow(y_list[i])
				# plt.show()

				# plt.imshow(gen_mask)
				# plt.show()
				
				# gen_mask = np.resize(gen_mask, (720, 1280))

				# from skimage.transform import resize as skresize
				# gen_mask = skresize(gen_mask, (720, 1280), anti_aliasing = True)

				gen_mask = cv2.resize(gen_mask, (1280, 720), interpolation = cv2.INTER_LINEAR)

				gen_mask[gen_mask >= 0.5] = 1.0
				gen_mask[gen_mask < 0.5] = 0.0

				gen_mask = (np.array(gen_mask)).astype('uint8')
				# print(gen_mask.shape, np.unique(gen_mask))

				current_frame_name = image_names[j][i][:-4] # Removing '.jpg' from filename
				
				abs_image_path = os.path.join(abs_video_path, current_frame_name + '.png')

				plt.imsave(abs_image_path, gen_mask)

		# -------------- End saving annotations --------------


		# print(sum(loss_list))
		loss_plot.append(sum(loss_list))
		epoch_avg_loss += sum(loss_list)
		iter_count += 1

	epoch_avg_loss /= num_batches
	print('avg loss:', epoch_avg_loss)

	if (epoch + 1) % SAVE_EVERY_N_EPOCH == 0:

		# for i, gen_mask in enumerate(yhat_list):
		# 	plt.imshow(y_list[i])
		# 	plt.show()

		# 	plt.imshow(gen_mask)
		# 	plt.show()
			
		# 	gen_mask[gen_mask >= 0.5] = 1.0
		# 	gen_mask[gen_mask < 0.5] = 0.0
		# 	plt.imshow(gen_mask)
		# 	plt.show()

		print('Saving ...')
		torch.save({
			'initializer': initializer.state_dict(),
			'encoder': encoder.state_dict(),
			'convlstmcell_decoder': convlstmcell_decoder.state_dict(),
			'convlstmcell_encoder': convlstmcell_encoder.state_dict(),
			'decoder': decoder.state_dict(),
			'optimizer': optimizer.state_dict(),
			'loss': epoch_avg_loss,
		}, os.path.join(save_models_dir, save_models_name + str(iter_count) + '.pt'))






# import matplotlib
# matplotlib.use('pdf')
# import matplotlib.pyplot as plt


# # Saving annotations
# print(len(yhat_list), yhat_list[0].shape)
# for i in range(yhat_list[0].shape[0]):
# 	print('i', i)
# 	current_video_name = name[i]
# 	abs_video_path = os.path.join(save_segmented_video_dir, current_video_name)
# 	try:
# 		os.makedirs(abs_video_path, exist_ok = True)
# 	except:
# 		pass

# 	for j in range(len(yhat_list)):
# 		print('j', j)

# 		gen_mask = yhat_list[j][i, :, :]
# 		# plt.imshow(y_list[i])
# 		# plt.show()

# 		# plt.imshow(gen_mask)
# 		# plt.show()
		
# 		gen_mask[gen_mask >= 0.5] = 1.0
# 		gen_mask[gen_mask < 0.5] = 0.0
# 		# gen_mask = np.resize(gen_mask, (720, 1280))

# 		# from skimage.transform import resize as skresize
# 		# gen_mask = skresize(gen_mask, (720, 1280), anti_aliasing = True)

# 		import cv2
# 		gen_mask = cv2.resize(gen_mask, (1280, 720), interpolation = cv2.INTER_NEAREST)
# 		gen_mask = (np.array(gen_mask)).astype('uint8')
# 		print(gen_mask.shape, np.unique(gen_mask))

# 		current_frame_name = image_names[j][i][:-4] # Removing '.jpg' from filename
		
# 		abs_image_path = os.path.join(abs_video_path, current_frame_name + '.png')

# 		plt.imsave(abs_image_path, gen_mask)

plt.plot(loss_plot)
plt.savefig(loss_plot_save_path)

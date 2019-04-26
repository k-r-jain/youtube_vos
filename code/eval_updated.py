import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import PIL

from valvos import ValidationYouTubeVOSLoader
from networks.initializer import Initializer
from networks.encoder import Encoder
from networks.convlstm import ConvLSTMCell
from networks.decoder import Decoder
from networks.nnutils import conv_unit
import copy
import sys

machine = 'home'
machine = 'newton'

batch_size = 1 # Very important because of the way data loader is written
fraction = 1.0
resize_dim = (256, 448)
final_width = 1280
final_height = 720
threshold = 0.5

mode = 'val'
device = 'cuda:0'
shuffle_data = True # Doesn't matter for validation anyway

print('-' * 50, 'EVAL UPDATED', '-' * 50)
if machine == 'home':
	saved_model_path = '/home/kartik/Desktop/newton_results/pa2_complete/vos_models_updated/vos_updated_4339.pt'
	root_data_dir = '/home/kartik/data/youtube_vos'
	save_segmented_video_dir = '/home/kartik/Desktop/validation_vos_results_updated'
	num_workers = 8

elif machine == 'newton':
	saved_model_path = '/home/cap6412.student4/pa2/vos_models_updated/vos_updated_4339.pt'
	root_data_dir = '/home/course.cap6412/youtubeVOS'
	save_segmented_video_dir = '/home/cap6412.student4/pa2/validation_vos_results_updated'
	num_workers = 4

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
image_transformation = transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])

mask_transformation = transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor()])

ytvos = ValidationYouTubeVOSLoader(root = root_data_dir, mode = mode, fraction = fraction, image_transformation = image_transformation, mask_transformation = mask_transformation)
data_loader = DataLoader(ytvos, batch_size = batch_size, shuffle = shuffle_data, num_workers = num_workers)

num_val_images = len(data_loader)


###### MODEL OBJECTS ######

encoded_h = int(resize_dim[0] / 32)
encoded_w = int(resize_dim[1] / 32)

initializer = Initializer().to(device)
encoder = Encoder().to(device)
convlstmcell_encoder = ConvLSTMCell(height = encoded_h, width = encoded_w).to(device)
convlstmcell_decoder = ConvLSTMCell(channels = 512, height = encoded_h, width = encoded_w).to(device)
decoder = Decoder(input_channels = 512, input_res = (encoded_h, encoded_w), output_res = resize_dim).to(device)
h_convolution = conv_unit(in_ch = 512 * 2, out_ch = 512, kernel_size = 3, padding = 1, activation = 'relu', batch_norm = True).to(device)


checkpoint = torch.load(saved_model_path)
initializer.load_state_dict(checkpoint['initializer'])
encoder.load_state_dict(checkpoint['encoder'])
convlstmcell_encoder.load_state_dict(checkpoint['convlstmcell_encoder'])
convlstmcell_decoder.load_state_dict(checkpoint['convlstmcell_decoder'])
decoder.load_state_dict(checkpoint['decoder'])
h_convolution.load_state_dict(checkpoint['h_convolution'])

initializer.eval()
encoder.eval()
convlstmcell_encoder.eval()
convlstmcell_decoder.eval()
decoder.eval()
h_convolution.eval()
###########################


try:
	os.makedirs(save_segmented_video_dir, exist_ok = True)
except:
	pass

for i, sample in enumerate(data_loader):

	video = sample['x'].to(device)
	mask_video = sample['y'].to(device)
	video_len = sample['t']
	video_name = sample['name']
	image_names = sample['image_names']
	image_names = [i[0] for i in image_names] # To convert to sane format: simple list
	mask_names = sample['mask_names']
	num_objects_from_metadata = sample['num_objects']
	image_names_from_metadata = sample['image_names_from_metadata']
	image_names_from_metadata = [i[0] for i in image_names_from_metadata]
	# print(video_name, video_len, image_names, mask_names, num_objects_from_metadata)

	abs_video_path = os.path.join(save_segmented_video_dir, video_name[0])
	try:
		os.makedirs(abs_video_path, exist_ok = True)
	except:
		pass
	
	total_unique_objects = 0
	corresponding_objects_mask_image_names = []
	corresponding_objects_mask_ids = []
	corresponding_objects_mask_pixel_values = []
	num_masks = mask_video.size(1)
	for mask_id in range(num_masks):
		current_mask = mask_video[:, mask_id, :, :, :].view(resize_dim[0], resize_dim[1])

		unique_vals = torch.unique(current_mask)
		for current_unique_val in unique_vals:
			if (current_unique_val != 0.0) and (current_unique_val not in corresponding_objects_mask_pixel_values):
				corresponding_objects_mask_pixel_values.append(current_unique_val.item())
				corresponding_objects_mask_ids.append(mask_id)
				corresponding_objects_mask_image_names.append(mask_names[mask_id][0])
				total_unique_objects += 1

	# print(video_name, corresponding_objects_mask_pixel_values, corresponding_objects_mask_image_names)
	if total_unique_objects != num_objects_from_metadata:
		print('Failure. Mismatch between metadata and seg masks. Abort.', video_name)
		break

	num_objects = total_unique_objects
	print(video_name, num_objects)
	final_predicted_mask_video_with_objects_superimposed = torch.zeros(1, video.size(1), 1, resize_dim[0], resize_dim[1])

	print(corresponding_objects_mask_ids, corresponding_objects_mask_pixel_values, corresponding_objects_mask_image_names)
	for current_object_id in range(num_objects):
		current_object_mask_id = corresponding_objects_mask_ids[current_object_id]
		current_object_pixel_value = corresponding_objects_mask_pixel_values[current_object_id]
		current_object_image_name = corresponding_objects_mask_image_names[current_object_id]
		# print('curr object magic num', current_object_mask_id, current_object_pixel_value, current_object_image_name)
		mask_with_currect_object = mask_video.clone()
		mask_with_currect_object = mask_with_currect_object[:, current_object_mask_id, :, :, :].view(resize_dim[0], resize_dim[1])

		# Keeping only one object and rescaling to {0.0, 1.0} for every forward pass
		# print('mask curr object', torch.unique(mask_with_currect_object))
		mask_with_currect_object[mask_with_currect_object != current_object_pixel_value] = 0.0
		final_mask = mask_with_currect_object / current_object_pixel_value
		# Convert to original shape for batch processing
		final_mask = final_mask.view(1, 1, 1, resize_dim[0], resize_dim[1])


		# plt.imshow(final_mask.view(resize_dim[0], resize_dim[1]))
		# plt.show()

		# Selecting part of video with first frame where the object appears
		edited_imagename = current_object_image_name[:-4] + '.jpg' # To remove '.png' and add '.jpg'
		# print(image_names)
		video_start_frame_index = image_names.index(edited_imagename)

		final_video = video[:, video_start_frame_index: , :, :, :]

		# Make predictions for the initial frames where object is not present as bg (zeros)
		# Helps keep tensor sizes consistent. Adding gt as first frame with object
		if video_start_frame_index != 0:
			predicted_segments = torch.zeros(1, video_start_frame_index, 1, resize_dim[0], resize_dim[1])
			predicted_segments = torch.cat((predicted_segments, final_mask.cpu()), dim = 1)
		else:
			predicted_segments = final_mask.cpu()


		# All the frames in video
		ci, hi = torch.zeros((1, 512, encoded_h, encoded_w)).to(device), torch.zeros((1, 512, encoded_h, encoded_w)).to(device)
		for frame_id in range(final_video.size(1)):
			xi = final_video[:, frame_id, :, :, :]

			xi = encoder(xi)
			ci, hi = convlstmcell_encoder(xi, ci, hi)
		
		h_video = hi


		ci, hi = initializer(torch.cat((final_video[:, 0, :, :, :], final_mask[:, 0, :, :, :]), dim = 1))
		
		video_timesteps = final_video.size(1)
		for frame_id in range(1, video_timesteps):
			xi = final_video[:, frame_id, :, :, :]
		
			xi = encoder(xi)
			xi = torch.cat((xi, h_video), dim = 1)
			xi = h_convolution(xi)
			ci, hi = convlstmcell_decoder(xi, ci, hi)
			yhati = decoder(hi)
			
			yhati[yhati >= threshold] = 1.0
			yhati[yhati < threshold] = 0.0
			yhati = yhati.view(1, 1, 1, resize_dim[0], resize_dim[1])
			predicted_segments = torch.cat((predicted_segments, yhati.detach().cpu()), dim = 1)

			# plt.imshow(yhati.detach().cpu().view(resize_dim[0], resize_dim[1]))
			# plt.pause(1)
		
		# final_predicted_mask_video_with_objects_superimposed[(predicted_segments == 1.0) and (final_predicted_mask_video_with_objects_superimposed == 0.0)] = current_object_pixel_value
		# print('pred obj', torch.unique(predicted_segments), current_object_pixel_value)
		final_predicted_mask_video_with_objects_superimposed[predicted_segments == 1.0] = current_object_pixel_value
		# print('final so far', torch.unique(final_predicted_mask_video_with_objects_superimposed))

	# To rescale entire tensor to [0.0, 1.0]
	final_predicted_mask_video_with_objects_superimposed = final_predicted_mask_video_with_objects_superimposed / max(corresponding_objects_mask_pixel_values)

	print('final unique', torch.unique(final_predicted_mask_video_with_objects_superimposed))
	# print('meta frames', image_names_from_metadata)
	video_timesteps = video.size(1)
	for video_frame_id in range(video_timesteps):
		current_mask = final_predicted_mask_video_with_objects_superimposed[0, video_frame_id, 0, :, :]

		current_mask = np.array(PIL.Image.fromarray(current_mask.detach().cpu().numpy()).resize((final_width, final_height), resample = PIL.Image.BILINEAR))

		# Since bilinear introduces anti-aliasing effects around masks
		rescaled_pixel_values = np.array(corresponding_objects_mask_pixel_values) / max(corresponding_objects_mask_pixel_values)
		rescaled_pixel_values = np.append(rescaled_pixel_values, 0.0)
		print(rescaled_pixel_values, rescaled_pixel_values.shape)
		for h in range(current_mask.shape[0]):
			for w in range(current_mask.shape[1]):
				if current_mask[h, w] not in rescaled_pixel_values:
					difference = np.absolute(rescaled_pixel_values - current_mask[h, w])
					difference_index = np.where(difference == min(difference))
					# print(rescaled_pixel_values[difference_index])
					current_mask[h, w] = rescaled_pixel_values[difference_index][0] # Dont know why

		curr_image_name = image_names[video_frame_id][:-4]
		if curr_image_name in image_names_from_metadata:
			abs_image_path = os.path.join(abs_video_path, curr_image_name + '.png')
			assert current_mask.shape == (720, 1280)
			# print(current_mask.shape)
			# plt.imsave(abs_image_path, current_mask)
			current_mask *= 255.0
			im = PIL.Image.fromarray(current_mask.astype(np.uint8))
			im.save(abs_image_path, "PNG")
	

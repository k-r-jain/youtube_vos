import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import json

class ValidationYouTubeVOSLoader(Dataset):
	def __init__(self, root = 'data/youtube_vos', mode = 'val', fraction = 1.0, image_transformation = None, mask_transformation = None):
		super(ValidationYouTubeVOSLoader, self).__init__()
		self.root = os.path.abspath(root)
		self.mode = mode
		self.fraction = fraction
		self.image_transformation = image_transformation
		self.mask_transformation = mask_transformation

		if self.mode == 'train':
			self.data_dir = os.path.join(self.root, 'train')
		elif self.mode == 'val':
			self.data_dir = os.path.join(self.root, 'valid')

		self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
		self.target_dir = os.path.join(self.data_dir, 'Annotations')
		self.filenames = os.listdir(self.image_dir)
		self.num_samples = int(self.fraction * len(self.filenames))

		self.metadata = json.load(open(os.path.join(self.data_dir, 'meta.json')))
		# print(self.metadata)

		# START HERE

	def __len__(self):
		return self.num_samples
	

	def __getitem__(self, index):

		if self.mode == 'val':
			self.video_name = self.filenames[index]
			self.image_names = os.listdir(os.path.join(self.image_dir, self.video_name))
			self.image_names = sorted(self.image_names)
			self.target_names = os.listdir(os.path.join(self.target_dir, self.video_name))
			self.target_names = sorted(self.target_names)
			self.current_len = len(self.image_names)
			
			objects_in_video = self.metadata["videos"][self.video_name]["objects"]
			num_objects_in_video = len(objects_in_video)

			final_img_array = []
			for curr_object in range(num_objects_in_video):
				try:
					current_img_array = objects_in_video[str(curr_object+1)]['frames']
					for one_image in current_img_array:
						if one_image not in final_img_array:
							final_img_array.append(one_image)
				except:
					print('ERROR object', curr_object, 'not in metadata since it is small', objects_in_video)

			self.final_img_array = sorted(final_img_array)

			self.video = []
			self.segmented_video = []

			for frame_index in range(len(self.image_names)):
				filename = os.path.join(self.image_dir, self.video_name, self.image_names[frame_index])
				try:
					image = Image.open(filename)
					if self.image_transformation:
						image = self.image_transformation(image)
					self.video.append(image)

				except:
					print('IMAGE loading issue', self.video_name)
			
			self.video = torch.stack(self.video, dim = 0)
			
			for frame_index in range(len(self.target_names)):
				filename = os.path.join(self.target_dir, self.video_name, self.target_names[frame_index])
				try:
					segmented_image = Image.open(filename)
					if self.mask_transformation:
						segmented_image = self.mask_transformation(segmented_image)
					self.segmented_video.append(segmented_image)

				except:
					print('MASK loading issue', self.video_name)

			self.segmented_video = torch.stack(self.segmented_video, dim = 0)

			sample = {'x': self.video, 'y': self.segmented_video, 't': self.current_len, 'name': self.video_name, 'image_names': self.image_names, 'mask_names': self.target_names, 'num_objects': num_objects_in_video, 'image_names_from_metadata': self.final_img_array}
			return sample

		elif self.mode == 'train':
			raise NotImplementedError()


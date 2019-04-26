import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
# from torchvision.datasets import ImageFolder


class YouTubeVOSLoader(Dataset):
    def __init__(self, root = 'data/youtube_vos', mode = 'train', fraction = 1.0, image_transformation = None, mask_transformation = None, num_frames = 5):
        super(YouTubeVOSLoader, self).__init__()
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

        # To get max length of frames over the entire dataset
        # self.max_sequence_len = 0
        # for i in range(self.num_samples):
        #     self.video_name = self.filenames[i]
        #     self.image_names = os.listdir(os.path.join(self.image_dir, self.video_name))
        #     if len(self.image_names) > self.max_sequence_len:
        #         self.max_sequence_len = len(self.image_names)
        # print(self.max_sequence_len)

        # Setting it to hard 5 frames for now
        self.max_sequence_len = num_frames

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, index):

        if self.mode == 'train':
            self.video_name = self.filenames[index]
            self.image_names = os.listdir(os.path.join(self.image_dir, self.video_name))
            self.image_names = sorted(self.image_names)
            self.target_names = os.listdir(os.path.join(self.target_dir, self.video_name))
            self.target_names = sorted(self.target_names)
            self.current_len = len(self.image_names)
            # print(len(self.image_names), len(self.target_names))
            # print(self.video_name)
            # print(self.image_names, self.target_names)

            self.image_name_list = []

            self.video = []
            self.segmented_video = []

            actual_frames_in_video = len(self.image_names)
            num_frames_loaded = 0
            left_to_right = True
            frame_index = -1
            while num_frames_loaded != self.max_sequence_len:
                if left_to_right:
                    frame_index += 1
                else:
                    frame_index -= 1


                filename = os.path.join(self.image_dir, self.video_name, self.image_names[frame_index])
                try:
                    image = Image.open(filename)
                    num_frames_loaded += 1

                    # if actual_frames_in_video < self.max_sequence_len:
                    #     print(self.video_name)
                    #     plt.imshow(image)
                    #     plt.show()
                    
                    if self.image_transformation:
                        image = self.image_transformation(image)
                    self.video.append(image)
                    self.image_name_list.append(self.image_names[frame_index])
                    # print(image.size)
                    # print(len(image.getbands()))
                except:
                    print('IMAGE loading issue', self.video_name)
                
                if frame_index == (actual_frames_in_video - 1):
                    left_to_right = False
                elif frame_index == 0:
                    left_to_right = True


            # for filename in self.image_names:
            #     filename = os.path.join(self.image_dir, self.video_name, filename)
            #     try:
            #         image = Image.open(filename).convert('RGB')
            #         # plt.imshow(image)
            #         # plt.show()
            #         # print(image.size)
            #         # print(len(image.getbands()))
            #     except:
            #         print('loading issue')
            #     if self.transformation:
            #         image = self.transformation(image)
            #     # print(image.size())
            #     self.video.append(image)
            # # To make video max length
            # while len(self.video) < self.max_sequence_len:
            #     image = torch.zeros(3, 128, 128)
            #     image = transforms.ToPILImage()(image)
            #     if self.transformation:
            #         image = self.transformation(image)
            #     self.video.append(image)
            
            self.video = torch.stack(self.video, dim = 0)
            

            
            actual_frames_in_video = len(self.target_names)
            num_frames_loaded = 0
            left_to_right = True
            frame_index = -1

            is_random_object_selected = False # Keep selected object constant throughout the video
            object_selected_fg = 0.0
            while num_frames_loaded != self.max_sequence_len:
                if left_to_right:
                    frame_index += 1
                else:
                    frame_index -= 1


                filename = os.path.join(self.target_dir, self.video_name, self.target_names[frame_index])
                try:
                    segmented_image = Image.open(filename)
                    # plt.imshow(segmented_image)
                    # plt.show()
                    num_frames_loaded += 1

                    # if actual_frames_in_video < self.max_sequence_len:
                    #     print(self.video_name)
                    #     plt.imshow(segmented_image, cmap = 'gray')
                    #     plt.show()

                    if self.mask_transformation:
                        segmented_image = self.mask_transformation(segmented_image)

                        # print(segmented_image.size())
                        # if actual_frames_in_video < self.max_sequence_len:
                        # print(segmented_image.size(), 'max', segmented_image.max(0), 'min', segmented_image.min(0))
                        unique_vals = torch.unique(segmented_image)
                        # print('fg bg', unique_vals)

                        # Since some annotations only have background
                        if unique_vals.size(0) > 1:

                            if is_random_object_selected:
                                selected_fg = object_selected_fg
                            else:
                                # Remove background
                                unique_vals_fg = unique_vals[unique_vals != 0.0]
                                # print('fg', unique_vals_fg)
                                # Select random object from existing
                                # Older Pytorch returns float -_-
                                random_index = int(torch.randint(0, unique_vals_fg.size(0), size = (1,)).item())
                                # print(random_index)
                                selected_fg = unique_vals_fg[random_index].item()
                                # print('selected', selected_fg)
                                object_selected_fg = selected_fg

                            
                            # Make everything else background
                            segmented_image[segmented_image != selected_fg] = 0.0
                            # print('random', torch.unique(segmented_image))

                            # Scale to be binary exactly 
                            max_val = torch.max(segmented_image)
                            segmented_image = segmented_image / max_val
                            # print('final mask', torch.unique(segmented_image))

                    self.segmented_video.append(segmented_image)

                    
                    # print(image.size)
                    # print(len(image.getbands()))
                except:
                    print('MASK loading issue', self.video_name)
                
                if frame_index == (actual_frames_in_video - 1):
                    left_to_right = False
                elif frame_index == 0:
                    left_to_right = True


            # for filename in self.target_names:
            #     filename = os.path.join(self.target_dir, self.video_name, filename)
            #     try:
            #         segmented = Image.open(filename)
            #         # plt.imshow(segmented, cmap = 'gray')
            #         # plt.show()
            #         # print(segmented.size)
            #         # print(len(segmented.getbands()))

            #     except:
            #         print('loading issue')
            #     if self.transformation:
            #         segmented = self.transformation(segmented)
            #     # print(segmented.size())
            #     self.segmented_video.append(segmented)
            # # To make segment max length
            # while len(self.segmented_video) < self.max_sequence_len:
            #     segmented = torch.zeros(1, 128, 128)
            #     segmented = transforms.ToPILImage()(segmented)
            #     if self.transformation:
            #         segmented = self.transformation(segmented)
            #     self.segmented_video.append(segmented)
            # print(len(self.segmented_video))

            self.segmented_video = torch.stack(self.segmented_video, dim = 0)
            # print('-' * 30)

            sample = {'x': self.video, 'y': self.segmented_video, 't': self.current_len, 'name': self.video_name, 'image_names': self.image_name_list}
            return sample

        elif self.mode == 'val':
            raise NotImplementedError()
import cv2 as cv
import numpy as np
import kornia
import torch
from torchvision import transforms
from abc import ABC, abstractmethod

class ImageMatcher(ABC):
    @abstractmethod
    def get_correspondences(self, image1, image2):
        return


class LOFTRImageMatcher(ImageMatcher):
    def __init__(self, location, gpu_num=None, patch_size=1000):
        if gpu_num is None:
            self.device= torch.device('cpu')
        else:
            self.device= torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.image_matcher=kornia.feature.LoFTR(pretrained=location).to(self.device)
        self.patch_size=patch_size
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.patch_corners=None

    def get_correspondences(self, neon_image, action_image, neon_point=None):
        action_tensor, action_scaled2original = self._preprocess_image(action_image)
        neon_image, self.patch_corners = self._get_image_patch(neon_image, neon_point) if neon_point is not None else neon_image.copy()
        neon_tensor, neon_scaled2original = self._preprocess_image(neon_image)
        input_dict = {
            "image0": neon_tensor,
            "image1": action_tensor
        }
        for k in input_dict.keys():
            input_dict[k]=input_dict[k].to(self.device)
        with torch.inference_mode():
            correspondences = self.image_matcher(input_dict)
        for k in correspondences.keys():
            correspondences[k]=correspondences[k].cpu().numpy()
        correspondences=self._rescale_correspondences(correspondences, neon_scaled2original, action_scaled2original)
        correspondences['keypoints0']=correspondences['keypoints0']+self.patch_corners[0]
        return correspondences

    def _preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        scaled_image = cv.resize(image, (round(540*image.shape[1]/image.shape[0]), 540)) 
        ratio_scaled2image = (image.shape[1]/scaled_image.shape[1] , 
                            image.shape[0]/scaled_image.shape[0])
        scaled_image = self.transform(scaled_image) 
        scaled_image = torch.unsqueeze(scaled_image, dim=0)
        return scaled_image, ratio_scaled2image

    def _get_image_patch(self, image, point):
        patch_corners = self._get_patch_corners(self.patch_size, point, image.shape)
        x_min, y_min = min(patch_corners[:,0]), min(patch_corners[:,1])
        x_max, y_max = max(patch_corners[:,0]), max(patch_corners[:,1])
        image_patch = image[y_min:y_max,x_min:x_max,:]
        return image_patch, patch_corners
    
    @staticmethod
    def _get_patch_corners(self, patch_size, point,image_shape):
        if point[0] < patch_size//2:
            x_min = 0
            x_max = patch_size
        elif point[0] < image_shape[1] - patch_size//2:
            x_min = point[0] - patch_size//2
            x_max = point[0] + patch_size//2
        else:
            x_min = image_shape[1] - patch_size
            x_max = image_shape[1]
        if point[1] < patch_size/2:
            y_min = 0
            y_max = patch_size
        elif point[1] < image_shape[0] - patch_size//2:
            y_min = point[1] - patch_size//2
            y_max = point[1] + patch_size//2
        else:
            y_min = image_shape[0] - patch_size
            y_max = image_shape[0]
        return  np.array([[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]], dtype=np.int32)
    
    def _rescale_correspondences(self,correspondences, neon_ratio, action_ratio):
        correspondences['keypoints0'] = correspondences['keypoints0'] * neon_ratio
        correspondences['keypoints1'] = correspondences['keypoints1'] * action_ratio
        return correspondences



class ImageMatcherFactory:
    def __init__(self,image_matcher, image_matcher_parameters):
        if image_matcher.upper() == 'LOFTR':
            self.matcher = LOFTRImageMatcher(image_matcher_parameters['location'], image_matcher_parameters['gpu_num'], image_matcher_parameters['patch_size'])
        # does all the ifs, switchcases and param settings specific to each matcher to instantiate the desired image matcher

    def get_matcher(self):
        return self.matcher


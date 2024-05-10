import cv2 as cv
import numpy as np
import av
import kornia
import torch
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from pupil_labs.action_cam_mapper.utils import VideoHandler
from pupil_labs.action_cam_mapper.optic_flow import OpticFlowCalculator


class ImageMatcher(ABC):
    @abstractmethod
    def get_correspondences(self, image1, image2):
        return


class LOFTRImageMatcher(ImageMatcher):
    def _init_(self, location, gpu_num=None, patch_size=1000):
        if gpu_num is None:
            self.device= torch.device('cpu')
        else:
            self.device= torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.image_matcher=kornia.feature.LoFTR(pretrained=location).to(self.device)
        self.patch_size=patch_size

    def get_correspondences(self, neon_image, action_image, neon_point=None):
        neon_img = self._get_image_patch(neon_image, neon_point) if neon_point else neon_image.copy()
        neon_img = self._preprocess_image(neon_img)
        action_img = self._preprocess_image(action_image)
        input_dict = {
            "image0": torch.unsqueeze(self.transform(neon_img),dim=0), 
            "image1": torch.unsqueeze(self.transform(action_img),dim=0)
        }
        for k in input_dict.keys():
            input_dict[k]=input_dict[k].to(self.device)
        with torch.inference_mode():
            correspondences = self.image_matcher(input_dict)
        for k in self.correspondences.keys():
            correspondences[k]=correspondences[k].cpu().numpy()
        # correspondences need to be changed back to the right scale (LOFTR uses scaled images) and the coordinates in the image patch need to be shifted  to the original coord system of the original image
        return correspondences

    def _preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (round(540*image.shape[1]/image.shape[0]), 540)) 
        return image

    def _get_image_patch(self, image, point):
        if point[0] < self.patch_size/2:
            x_min = 0
            x_max = self.patch_size
        elif point[0] < image.shape[1] - self.patch_size/2:
            x_min = point[0] - self.patch_size/2
            x_max = point[0] + self.patch_size/2
        else:
            x_min = image.shape[1] - self.patch_size
            x_max = image.shape[1]
        if point[1] < self.patch_size/2:
            y_min = 0
            y_max = self.patch_size
        elif point[1] < image.shape[0] - self.patch_size/2:
            y_min = point[1] - self.patch_size/2
            y_max = point[1] + self.patch_size/2
        else:
            y_min = image.shape[0] - self.patch_size
            y_max = image.shape[0]
        image_patch = image[y_min:y_max,x_min:x_max,:]
        self.patch_corners = np.array([[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]], dtype=np.int32)
        return image_patch
    
    def _rescaling_correspondences(self,):
        pass

    

class ImageMatcherFactory:
    def __init__(self):
        pass
        self.matcher = None
        # does all the ifs, switchcases and param settings specific to each matcher to instantiate the desired image matcher

    def get_matcher(self):
        return self.matcher


class NeonToaction_cameraGazeMapper:
    def __init__(
        self,
        neon_gaze_csv,
        neon_video_dir,
        action_cameracamera_video_dir,
        neon_worldtimestamps,
        action_cameracamera_worldtimestamps,
        image_matcher,
        image_matcher_parameters,
        neon_opticflow_csv=None,
        action_camera_opticflow_csv=None,
    ) -> None:
        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # <- at 200Hz
        # maybe add a method in VideoHandler that sets the worldtimestamps to the relative timestamps (thinking it might also make the query of OF easier)
        self.neon_video = VideoHandler(neon_video_dir)
        self.action_camera_video = VideoHandler(action_cameracamera_video_dir)
        self.neon_worldtimestamps = pd.read_csv(neon_worldtimestamps)
        self.action_camera_worldtimestamps = pd.read_csv(
            action_cameracamera_worldtimestamps)
        self.neon_opticflow = pd.read_csv(neon_opticflow_csv)
        self.action_camera_opticflow = pd.read_csv(action_camera_opticflow_csv)
        # create an imageMatcher object with the specified method (LOFTR,DISK+LG,etc)
        self.image_matcher = ImageMatcherFactory(
            image_matcher, image_matcher_parameters).get_matcher()

    def map_gaze(self):
        for gaze_information in self.neon_gaze:
            if self._gaze_between_video_frames(gaze_information['world timestamp']):
                semantically_correct_gaze, neon_timestamp, action_camera_timestamp = self._adjust_gaze_with_optic_flow(
                    gaze_information['gaze x [px]', 'gaze y [px]'], gaze_information['world timestamp'])
            gaze_in_action_camera = self._map_one_gaze(
                semantically_correct_gaze, neon_timestamp, action_camera_timestamp)

    def _gaze_between_video_frames(self, gaze_timestamp):
        return max(self.neon_worldtimestamps[0], self.action_camera_worldtimestamps[0]) < gaze_timestamp < min(self.neon_worldtimestamps[-1], self.action_camera_worldtimestamps[-1])

    def _adjust_gaze_with_optic_flow(self, gaze_point, gaze_timestamp):
        closest_neon_timestamp, semantically_correct_gaze = self.move_point_to_video_timestamp(
            gaze_point, gaze_timestamp)
        closest_action_camera_timestamp = self.action_camera_video.get_closest_timestamp(
            closest_neon_timestamp)
        return semantically_correct_gaze, closest_neon_timestamp, closest_action_camera_timestamp

    def _map_one_gaze(self, gaze_coordinates, action_camera_timestamp, neon_timestamp):
        action_camera_frame = self._get_frame(action_camera_timestamp)
        neon_frame = self.neon_video.get_frame(neon_timestamp)
        correspondences = self.image_matcher.get_correspondences(
            neon_frame, action_camera_frame, gaze_coordinates)
        gaze_in_action_camera = self._estimate_transformation(
            correspondences, gaze_coordinates)
        return gaze_in_action_camera

    def _estimate_transformation(self, correspondences, point_to_be_transformed):
        self._filter_correspondences(correspondences)
        self.transformation = cv.estimateHomography(correspondences)
        transformed_point = cv.perspectiveTransform(
            point_to_be_transformed, self.transformation)
        return transformed_point

    def _filter_correspondences(self, correspondences):
        pass
        # filter the correspondences based on the method used
        # return the filtered correspondences

    def move_point_to_video_timestamp(self,
                                      point_coordinates,
                                      point_timestamp):
        pass
    # moves a point in an arbitrary timestamp to the closest video frame along the optic flow displacement
    # point gets moved to the closest timestamp between the optic flow timestamps
        # previous_video_timestamp, next_video_timestamp = get surrounding video timestamps
        # opticflow_displacement_between_frames = self.getOpticFlow(previous_video_timestamp, next_video_timestamp, 'displacement')
        # dx_dx_dt = opticflow_displacement_between_frames/(next_video_timestamp - previous_video_timestamp)
        # closest_timestamps = get the closest video timestamp to point_timestamp
        # dx_dy = dx_dx_dt * (closest_timestamp-gaze_timestamp)
        # return closest_timestamp, point_coordinates + dx_dy
        # returns the video timestamp to which the point was moved and the new point coordinates

    def move_point_to_arbitrary_timestamp(self,
                                          point_coordinates,
                                          video_timestamp,
                                          target_timestamp):
        pass
    # moves a point found in a timestamp of the video to an arbitrary timestamp
    # check if the target_timestamp and video_timestamp are in the video
    # if target_timestamp > video_timestamp # target timestamp is in the future with respect to the video timestamp
        # opticflow_displacement_between_frames = obtain optic flow between video_timestamp and video_timestamp + 1
        # dx_dx_dt = opticflow_displacement_between_frames/(video_timestamp + 1 - video_timestamp)
    # if target_timestamp < video_timestamp # target timestamp is in the past with respect to the video timestamp
        # opticflow_displacement_between_frames = obtain optic flow between video_timestamp and video_timestamp - 1
        # dx_dx_dt = opticflow_displacement_between_frames/(video_timestamp - video_timestamp-1)
    # dx_dy = dx_dx_dt * (target_timestamp - video_timestamp)
    # return target_timestamp, point_coordinates + dx_dy
    # returns the target timestamp and the new point coordinates

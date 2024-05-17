import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from pupil_labs.action_cam_mapper.utils import VideoHandler
from pupil_labs.action_cam_mapper.feature_matcher import ImageMatcherFactory


class ActionCameraGazeMapper:
    def __init__(
        self,
        neon_gaze_csv,
        neon_video_dir,
        action_video_dir,
        neon_worldtimestamps,
        action_worldtimestamps,
        image_matcher,
        image_matcher_parameters,
        neon_opticflow_csv=None,
        action_opticflow_csv=None,
    ) -> None:
        self.neon_video = VideoHandler(neon_video_dir)
        self.action_video = VideoHandler(action_video_dir)
        self.neon_worldtimestamps = pd.read_csv(neon_worldtimestamps)
        self.action_worldtimestamps = pd.read_csv(
            action_worldtimestamps)
        self.action2neon_offset = self.action_worldtimestamps['timestamp [ns]'].values[0] - self.neon_worldtimestamps['timestamp [ns]'].values[0]
        self.action2neon_offset/=1e9
        self.neon_opticflow = pd.read_csv(neon_opticflow_csv)
        self.action_opticflow = pd.read_csv(action_opticflow_csv)
        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # <- at 200Hz
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
        return max(self.neon_worldtimestamps[0], self.action_worldtimestamps[0]) < gaze_timestamp < min(self.neon_worldtimestamps[-1], self.action_worldtimestamps[-1])

    def _adjust_gaze_with_optic_flow(self, gaze_point, gaze_timestamp):
        closest_neon_timestamp = self.neon_video.get_closest_timestamp((gaze_timestamp-self.neon_worldtimestamps['timestamp [ns]'].values[0])/1e9)
        semantically_correct_gaze = self.move_point_to_video_timestamp(
            gaze_point, gaze_timestamp, closest_neon_timestamp, self.neon_opticflow)
        closest_action_camera_timestamp = self.action_video.get_closest_timestamp(
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
                                      point_timestamp,
                                      opticflow_timestamp,
                                      opticflow):
        pass
    # moves a point in an arbitrary timestamp to the closest video frame along the optic flow displacement
    # point gets moved to the closest timestamp between the optic flow timestamps
        # previous_video_timestamp, next_video_timestamp = get surrounding video timestamps
        # opticflow_displacement_between_frames = opticFlow(previous_video_timestamp, next_video_timestamp, 'displacement')
        # dx_dx_dt = opticflow_displacement_between_frames/(next_video_timestamp - previous_video_timestamp)
        # closest_timestamps = get the closest video timestamp to point_timestamp
        # dx_dy = dx_dx_dt * (closest_timestamp-gaze_timestamp)
        return point_coordinates + dx_dy
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

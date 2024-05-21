import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from utils import VideoHandler
from feature_matcher import ImageMatcherFactory


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
        self.neon_opticflow = pd.read_csv(neon_opticflow_csv,dtype={'start': np.float32, 'end': np.float32, 'avg_displacement_x': np.float32, 'avg_displacement_y': np.float32, 'angle': np.float32})
        self.action_opticflow = pd.read_csv(action_opticflow_csv,dtype={'start': np.float32, 'end': np.float32, 'avg_displacement_x': np.float32, 'avg_displacement_y': np.float32, 'angle': np.float32})
        self.image_matcher = ImageMatcherFactory(
            image_matcher, image_matcher_parameters).get_matcher()
        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # <- at 200Hz
        self.action_gaze = self._initialize_action_gaze()
        

    def _initialize_action_gaze(self):
        # creates the action_gaze dataframe with the same columns as the neon_gaze dataframe
        action_dataframe = pd.DataFrame.from_dict({col:[None for _ in self.neon_gaze['timestamp [ns]'].values] for col in self.neon_gaze.columns})
        action_dataframe.loc[:,['section id','recording id','timestamp [ns]','worn','fixation id','blink id']] = self.neon_gaze[['section id','recording id','timestamp [ns]','worn','fixation id','blink id']].values
        # it also limits the action_gaze dataframe to the timestamps of the action video
        pass

    def map_gaze(self, saving_path=None):
        for gaze_world_ts in self.action_gaze['timestamp [ns]'].values:
            #TODO: add a progress bar
            gaze_neon=self.neon_gaze.loc[self.neon_gaze['timestamps [ns]']==gaze_world_ts,['gaze x [px]', 'gaze y [px]']].values.reshape(1,2)
            if self._gaze_between_video_frames(gaze_world_ts):
                semantically_correct_gaze, neon_timestamp, action_timestamp = self._adjust_gaze_with_optic_flow(gaze_neon,
                    gaze_world_ts)
                gaze_action_camera = self._map_one_gaze(
                    semantically_correct_gaze, neon_timestamp, action_timestamp)
                gaze_action_camera = self._move_point_to_arbitrary_timestamp(gaze_action_camera,action_timestamp,self.action_opticflow,gaze_world_ts)
            else:
                gaze_relative_timestamp = (gaze_world_ts - self.neon_worldtimestamps['timestamp [ns]'].values[0])/1e9
                neon_timestamp = self.neon_video.get_closest_timestamp(gaze_relative_timestamp)
                action_timestamp = self.action_video.get_closest_timestamp(neon_timestamp-self.action2neon_offset)
                gaze_action_camera = self._map_one_gaze(
                    gaze_neon, neon_timestamp, action_timestamp)
            self.action_gaze.loc[self.action_gaze['timestamp [ns]']==gaze_world_ts, ['gaze x [px]', 'gaze y [px]']] = gaze_action_camera
        if saving_path is None:
            self.action_gaze.to_csv(Path(self.neon_video.video_dir).parent/'action_gaze.csv', index=False)
        else:
            self.action_gaze.to_csv(saving_path, index=False)

    def _gaze_between_video_frames(self, gaze_timestamp):
        return max(self.neon_worldtimestamps['timestamp [ns]'].values[0], self.action_worldtimestamps['timestamp [ns]'].values[0]) < gaze_timestamp < min(self.neon_worldtimestamps['timestamp [ns]'].values[-1], self.action_worldtimestamps['timestamp [ns]'].values[-1])

    def _adjust_gaze_with_optic_flow(self, gaze_point, gaze_timestamp):
        gaze_relative_timestamp = (gaze_timestamp - self.neon_worldtimestamps['timestamp [ns]'].values[0])/1e9
        closest_neon_timestamp = self.neon_video.get_closest_timestamp(gaze_relative_timestamp)
        semantically_correct_gaze = self._move_point_to_video_timestamp(
            gaze_point, gaze_relative_timestamp, closest_neon_timestamp, self.neon_opticflow)
        closest_action_timestamp = self.action_video.get_closest_timestamp(
            closest_neon_timestamp-self.action2neon_offset)
        return semantically_correct_gaze, closest_neon_timestamp, closest_action_timestamp

    def _map_one_gaze(self, gaze_coordinates, action_timestamp, neon_timestamp):
        action_camera_frame = self.action_video.get_frame_by_timestamp(action_timestamp)
        neon_frame = self.neon_video.get_frame_by_timestamp(neon_timestamp)
        if np.all(neon_frame==100):
            return gaze_coordinates
        correspondences = self.image_matcher.get_correspondences(
            neon_frame, action_camera_frame, gaze_coordinates)
        correspondences, _ = self._filter_correspondences(correspondences, gaze_coordinates, neon_frame.shape)
        gaze_in_action_camera = self._estimate_transformation(
            correspondences, gaze_coordinates)
        return gaze_in_action_camera

    def _estimate_transformation(self, correspondences, point_to_be_transformed):
        #TODO: Add here add an exception to see if there are very close correspondences to the  point_to_be_transformed 
        neon_pts = np.float32(correspondences['keypoints0']).reshape(-1, 1, 2)
        action_pts = np.float32(correspondences['keypoints1']).reshape(-1, 1, 2)
        self.transformation, mask = cv.findHomography(neon_pts, action_pts, cv.RANSAC, 5.0)
        point_to_be_transformed = np.float32(point_to_be_transformed).reshape(-1, 1, 2)
        transformed_point = cv.perspectiveTransform(
            point_to_be_transformed, self.transformation)
        return transformed_point.reshape(1,2)

    def _filter_correspondences(self, correspondences, point_to_be_transformed, image_shape):
        prev_patch_size = self.image_matcher.patch_size
        prev_patch_corners = self.image_matcher.patch_corners
        while True:
            new_patch_size = prev_patch_size - 100
            new_patch_corners = self.image_matcher._get_patch_corners(new_patch_size, point_to_be_transformed, image_shape)
            kept_kp_index=[]
            for i, kp in enumerate(correspondences['keypoints0']):
                if min(new_patch_corners[:,0])<kp[0]<max(new_patch_corners[:,0]) and min(new_patch_corners[:,1])<kp[1]<max(new_patch_corners[:,1]):
                    kept_kp_index.append(i)
            if len(kept_kp_index) < 100: 
                return correspondences, prev_patch_corners
            for k in correspondences.keys():
                correspondences[k]=correspondences[k][kept_kp_index]
            prev_patch_corners = new_patch_corners
            prev_patch_size = new_patch_size
            if new_patch_size == 100:
                return correspondences, prev_patch_corners

    def _move_point_to_video_timestamp(self,
                                    point_coordinates,
                                    point_timestamp,
                                    opticflow_timestamp,
                                    opticflow):
        """Moves a point in an arbitrary timestamp to the closest video frame along the given optic flow.

        Args:
            point_coordinates (ndarray): 2D coordinates of the point to be moved
            point_timestamp (float): Timestamp of the point to be moved
            opticflow_timestamp (float): Timestamp of the closest frame of the optic flow signal
            opticflow (DataFrame): Optic Flow signal of the video

        Returns:
            ndarray: New coordinates of the point after being moved
        """
        time_difference=opticflow_timestamp-point_timestamp 
        if time_difference==0:
            return point_coordinates
        elif time_difference>0: # The point is in the past, so it needs to be moved to the 'future'
            opticflow_displacement_between_frames = opticflow.loc[opticflow['end'] == opticflow_timestamp,['avg_displacement_x','avg_displacement_y']].values
            dx_dy_dt = opticflow_displacement_between_frames/np.diff(opticflow.loc[opticflow['end'] == opticflow_timestamp,['start','end']].values)
        elif time_difference<0: #The point is in the future, so it needs to be moved to the 'past', against the optic flow between the optic flow timestamp and the next timestamp
            opticflow_displacement_between_frames = opticflow.loc[opticflow['start'] == opticflow_timestamp,['avg_displacement_x','avg_displacement_y']].values
            dx_dy_dt = opticflow_displacement_between_frames/np.diff(opticflow.loc[opticflow['start'] == opticflow_timestamp,['start','end']].values)
        dx_dy = dx_dy_dt * time_difference
        return point_coordinates + dx_dy

    def _move_point_to_arbitrary_timestamp(self,
                                        point_coordinates,
                                        video_timestamp,
                                        opticflow,
                                        target_timestamp):
        gaze_relative_timestamp = (target_timestamp - self.neon_worldtimestamps['timestamp [ns]'].values[0])/1e9
        aligned_video_timestamp = video_timestamp + self.action2neon_offset
        time_difference = gaze_relative_timestamp - aligned_video_timestamp
        if time_difference == 0:
            return point_coordinates
        elif time_difference > 0: # target timestamp is in the future with respect to the video timestamp
            opticflow_displacement_between_frames =opticflow.loc[opticflow['start'] == video_timestamp,['avg_displacement_x','avg_displacement_y']].values
            dx_dy_dt = opticflow_displacement_between_frames/np.diff(opticflow.loc[opticflow['start'] == video_timestamp,['start','end']].values)
        elif time_difference < 0: # target timestamp is in the past with respect to the video timestamp
            opticflow_displacement_between_frames = opticflow.loc[opticflow['end'] == video_timestamp,['avg_displacement_x','avg_displacement_y']].values
            dx_dy_dt = opticflow_displacement_between_frames/np.diff(opticflow.loc[opticflow['end'] == video_timestamp,['start','end']].values)
        dx_dy = dx_dy_dt * time_difference
        return point_coordinates + dx_dy

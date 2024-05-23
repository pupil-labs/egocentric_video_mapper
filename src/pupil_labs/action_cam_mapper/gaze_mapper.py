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
        patch_size=1000,
    ) -> None:
        self.neon_video = VideoHandler(neon_video_dir) #name consistency
        self.action_video = VideoHandler(action_video_dir)
        self.neon_worldtimestamps = pd.read_csv(neon_worldtimestamps)
        self.action_worldtimestamps = pd.read_csv(
            action_worldtimestamps)
        self.action2neon_offset = self.action_worldtimestamps['timestamp [ns]'].values[0] - self.neon_worldtimestamps['timestamp [ns]'].values[0]
        self.action2neon_offset/=1e9
        self.neon_opticflow = pd.read_csv(neon_opticflow_csv,dtype=np.float32)
        self.action_opticflow = pd.read_csv(action_opticflow_csv,dtype=np.float32)
        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # <- at 200Hz
        self.action_gaze = self._create_action_gaze()
        self.image_matcher = ImageMatcherFactory(
            image_matcher, image_matcher_parameters).get_matcher()
        self.patch_size = patch_size
        self.transformation = np.array([[self.action_video.width/self.neon_video.width, 0, 0], 
                                        [0, self.action_video.height/self.neon_video.height, 0],
                                        [0, 0, 1]], dtype=np.float32)

    def _create_action_gaze(self):
        """Creates a DataFrame with the same formatting as the neon_gaze DataFrame, the 'gaze x [px]',
        'gaze y [px]', 'azimuth [deg]' and 'elevation [deg]' columns are filled with None values, while the rest of the columns keep the same values as the neon_gaze DataFrame.

        Returns:
            DataFrame: A DataFrame with the same formatting as the neon_gaze DataFrame
        """
        action_dataframe = pd.DataFrame.from_dict({col:[None for _ in self.neon_gaze['timestamp [ns]'].values] for col in self.neon_gaze.columns})
        action_dataframe.loc[:,['section id','recording id','timestamp [ns]','worn','fixation id','blink id']] = self.neon_gaze[['section id','recording id','timestamp [ns]','worn','fixation id','blink id']].values
        action_dataframe = action_dataframe[action_dataframe['timestamp [ns]'] >= self.action_worldtimestamps['timestamp [ns]'].values[0]]
        action_dataframe = action_dataframe[action_dataframe['timestamp [ns]'] <= self.action_worldtimestamps['timestamp [ns]'].values[-1]]
        return action_dataframe

    def map_gaze(self, saving_path=None):

        for i, gaze_world_ts in enumerate(self.action_gaze['timestamp [ns]'].values):
            print(i)
            gaze_neon=self.neon_gaze.loc[self.neon_gaze['timestamp [ns]']==gaze_world_ts,['gaze x [px]', 'gaze y [px]']].values.reshape(1,2)
            gaze_relative_timestamp = (gaze_world_ts - self.neon_worldtimestamps['timestamp [ns]'].values[0])/1e9
            neon_timestamp = self.neon_video.get_closest_timestamp(gaze_relative_timestamp)
            action_timestamp = self.action_video.get_closest_timestamp(neon_timestamp-self.action2neon_offset)

            if self._gaze_between_video_frames(gaze_world_ts):
                semantically_correct_gaze = self._move_point_to_video_timestamp(gaze_neon, gaze_relative_timestamp, neon_timestamp, self.neon_opticflow)
                gaze_action_camera = self._map_one_gaze(
                    semantically_correct_gaze, neon_timestamp, action_timestamp)
                gaze_action_camera = self._move_point_to_arbitrary_timestamp(gaze_action_camera,action_timestamp,self.action_opticflow,gaze_world_ts)
            else:
                gaze_action_camera = self._map_one_gaze(
                    gaze_neon, neon_timestamp, action_timestamp)
            
            print(f'Gaze ({gaze_neon}) at {gaze_world_ts} mapped to {gaze_action_camera}')
            self.action_gaze.loc[self.action_gaze['timestamp [ns]']==gaze_world_ts, ['gaze x [px]', 'gaze y [px]']] = gaze_action_camera
        
        if saving_path is None:
            self.action_gaze.to_csv(Path(self.neon_video.video_dir).parent/'action_gaze.csv', index=False)
        else:
            self.action_gaze.to_csv(saving_path, index=False)

    def _gaze_between_video_frames(self, gaze_timestamp):
        return max(self.neon_worldtimestamps['timestamp [ns]'].values[0], self.action_worldtimestamps['timestamp [ns]'].values[0]) < gaze_timestamp < min(self.neon_worldtimestamps['timestamp [ns]'].values[-1], self.action_worldtimestamps['timestamp [ns]'].values[-1])

    def _map_one_gaze(self, gaze_coordinates, neon_timestamp,action_timestamp):
        action_frame = self.action_video.get_frame_by_timestamp(action_timestamp)
        neon_frame = self.neon_video.get_frame_by_timestamp(neon_timestamp)
        if np.all(neon_frame==100):
            print(f'Neon frame at {neon_timestamp} is all gray')
            return gaze_coordinates
        patch_corners = self._get_patch_corners(self.patch_size, gaze_coordinates, neon_frame.shape)
        correspondences = self.image_matcher.get_correspondences(
            neon_frame, action_frame, patch_corners)
        correspondences, new_patch_corners = self._filter_correspondences(correspondences.copy(), gaze_coordinates, neon_frame.shape)
        print(f'Number of correspondences: {len(correspondences["keypoints0"])} at {abs(new_patch_corners[0,0]-new_patch_corners[2,0])} patch size')
        gaze_in_action_camera=self._transform_point(correspondences,gaze_coordinates)
        return gaze_in_action_camera

    def _transform_point(self, correspondences, point_to_be_transformed):
        self._estimate_transformation(correspondences)
        point_to_be_transformed = np.float32(point_to_be_transformed).reshape(-1, 1, 2)
        transformed_point = cv.perspectiveTransform(
            point_to_be_transformed, self.transformation)
        return transformed_point.reshape(1,2)
    
    def _estimate_transformation(self, correspondences):
        # returns callable
        neon_pts = np.float32(correspondences['keypoints0']).reshape(-1, 1, 2)
        action_pts = np.float32(correspondences['keypoints1']).reshape(-1, 1, 2)
        prev_transformation = self.transformation
        try:
            self.transformation, mask = cv.findHomography(neon_pts, action_pts, cv.RANSAC, 5.0)
            if mask.ravel().sum() ==0:
                print('Not enough inliers, using previous transformation')
                # may be better to not do this, rather just not map the gaze and leave it empty
                self.transformation = prev_transformation
        except cv.error:
            print('Homography could not be estimated, using previous transformation')

    def _filter_correspondences(self, correspondences, point_to_be_transformed, image_shape):
        prev_patch_size = self.patch_size
        prev_patch_corners = self._get_patch_corners(prev_patch_size, point_to_be_transformed, image_shape)
        while True:
            new_patch_size = prev_patch_size - 100
            new_patch_corners = self._get_patch_corners(new_patch_size, point_to_be_transformed, image_shape)
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
    
    @staticmethod
    def _get_patch_corners(patch_size,point,image_shape):
        point = point.reshape(-1)
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

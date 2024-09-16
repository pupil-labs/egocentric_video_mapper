import logging
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from pupil_labs.egocentric_video_mapper.feature_matcher import get_matcher
from pupil_labs.egocentric_video_mapper.video_handler import VideoHandler


class EgocentricMapper:
    def __init__(
        self,
        neon_gaze_csv,
        neon_video_path,
        alternative_video_path,
        neon_timestamps,
        alternative_timestamps,
        image_matcher,
        image_matcher_parameters,
        neon_opticflow_csv=None,
        alternative_opticflow_csv=None,
        output_dir=None,
        patch_size=1000,
        alternative_fov=[145, 76],
        logging_level="INFO",
    ):
        """Class to map gaze from the Neon scene camera to an alternative camera. The gaze is mapped by calling the map_gaze method which uses correspondences between the two cameras to calculate the gaze transformation.

        Args:
            neon_gaze_csv (str): Path to the gaze.csv file from the Neon recording. The file should come from the Timeseries Data + Scene Video download from Pupil Cloud.
            neon_video_path (str): Path to the video recorded by the Neon scene camera. The video comes from the Timeseries Data + Scene Video download from Pupil Cloud.
            alternative_video_path (str): Path to the video recorded by the alternative camera.
            neon_timestamps (str): Path to the world_timestamps.csv from the Neon recording. The file should come from the Timeseries Data + Scene Video download from Pupil Cloud.
            alternative_timestamps (str): Path to the alternative_camera_timestamps.csv from the alternative camera. This file is create by utils.write_timestamp_csv
            image_matcher (str): Name of the image matcher to be used. For already implemented matcher check the feature_matcher module.
            image_matcher_parameters (dict): Set of specific parameters for the image matcher. The parameters are passed as a dictionary with the parameter name as the key and the parameter value as the value.
            neon_opticflow_csv (str, optional): Path to the optic flow pertaining to the Neon scene video. Used for semantic alignment of the gaze signal. If no file is provided no semantic aligment takes place. Defaults to None.
            alternative_opticflow_csv (str, optional): Path to the optic flow pertaining to the alternative camera video. Used for semantic alignment of the gaze signal. If no file is provided no semantic aligment takes place. Defaults to None.
            output_dir (str, optional): Path to the directory to store the mapped gaze, if None is given the mapped gaze is saved in the same directory as the neon_gaze_csv. Defaults to None.
            patch_size (int, optional): Size of the context image window to take around any given gaze. Defaults to 1000.
            alternative_fov (list, optional): Field of view in degrees of the Neon Scene camera (can be 2D or 1D). If 1D, it is assumed that the camera has the same field of view in both axes. If 2D, it is assumed that [fov_x, fov_y]. Defaults to [145, 76].
            logging_level (str, optional): Level of logging to be used. Defaults to "ERROR".
        """
        self.neon_video = VideoHandler(neon_video_path)
        self.alt_video = VideoHandler(alternative_video_path)

        self.neon_vid_ts_nanosec = pd.read_csv(neon_timestamps)
        self.alt_vid_ts_nanosec = pd.read_csv(alternative_timestamps)
        self.alt2neon_offset_sec = (
            self.alt_vid_ts_nanosec["timestamp [ns]"].values[0]
            - self.neon_vid_ts_nanosec["timestamp [ns]"].values[0]
        ) / 1e9

        self.neon_opticflow = pd.read_csv(neon_opticflow_csv, dtype=np.float32)
        self.alt_opticflow = pd.read_csv(alternative_opticflow_csv, dtype=np.float32)

        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # @ 200Hz
        self.alt_gaze = self._create_alternative_gaze_df()

        self.image_matcher = get_matcher(image_matcher, image_matcher_parameters)

        self.output_dir = Path(output_dir or Path(neon_video_path).parent)

        self.corresponding_alt_ts_idx = self._get_corresponding_timestamps_index(
            self.alt_vid_ts_nanosec["timestamp [ns]"].values,
            self.alt_gaze["timestamp [ns]"].values,
        )
        self.corresponding_neon_ts_idx = self._get_corresponding_timestamps_index(
            self.neon_vid_ts_nanosec["timestamp [ns]"].values,
            self.alt_gaze["timestamp [ns]"].values,
        )

        self.patch_size = patch_size
        self.gaze_transformation = np.array(
            [
                [self.alt_video.width / self.neon_video.width, 0, 0],
                [0, self.alt_video.height / self.neon_video.height, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.correspondences = None
        self.alt_fov = np.asarray(alternative_fov)
        # Reported values in Neon documentation https://docs.pupil-labs.com/neon/data-collection/data-streams/#scene-video
        self.neon_fov = np.asarray([103, 77])

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def map_gaze(
        self,
        saving_path=None,
        refresh_time_thrshld=None,
        optic_flow_thrshld=None,
        gaze_change_thrshld=None,
    ):
        """
        Args:
            saving_path (str, optional): Saving path for the mapped gaze, in format /path/to/alternative_gaze.csv, if None is given it saves the mapped gaze as 'alternative_camera_gaze.csv' in the output directory specified in the initialization of the object. Defaults to None.
            refresh_time_thrshld (int, optional): Maximum allowed time elapsed, in seconds, since the last computation of image correspondences. If set to None this threshold is not enforced. Defaults to None.
            opticf_thrshld (int, optional): Maximum allowed cummulative optic flow, in pixels, since the last computation of image correspondences. If set to None this threshold is not enforced Defaults to None.
            gaze_change_thrshld (int, optional): Maximum allowed neon gaze change, in  pixels, since the last computation of image correspondences. If set to None this threshold is not enforced. Defaults to None.
        """
        refresh_time_thrshld = (
            refresh_time_thrshld * 200 if refresh_time_thrshld is not None else None
        )
        neon_gaze_df = pd.DataFrame(
            {
                ("timestamp_nanosec", ""): self.neon_gaze["timestamp [ns]"],
                ("gaze", "x"): self.neon_gaze["gaze x [px]"],
                ("gaze", "y"): self.neon_gaze["gaze y [px]"],
            }
        )
        gazes_since_refresh = 0
        acc_alt_opticflow = 0
        acc_neon_opticflow = 0
        last_gaze = neon_gaze_df.gaze.values[0]

        for gaze_idx, gaze_ts in enumerate(
            tqdm(
                self.alt_gaze["timestamp [ns]"].values,
                desc="Mapping gaze signal",
            )
        ):

            gaze_neon = neon_gaze_df[
                neon_gaze_df.timestamp_nanosec == gaze_ts
            ].gaze.values.reshape(1, 2)

            gaze_rel_ts, neon_rel_ts, alt_rel_ts = self._obtain_relative_ts(
                gaze_ts, gaze_idx
            )
            self.logger.info(f"({gaze_idx}) Transforming gaze {gaze_neon} at {gaze_ts}")

            if self._ts_between_video_frames(gaze_ts):
                gaze_neon = self._move_point_to_video_timestamp(
                    gaze_neon,
                    gaze_rel_ts,
                    neon_rel_ts,
                    self.neon_opticflow,
                )

            # check if new frames need to be retrieved
            if gaze_idx == 0 or (
                self.corresponding_neon_ts_idx[gaze_idx]
                != self.corresponding_neon_ts_idx[gaze_idx - 1]
            ):
                neon_frame, neon_opticflow = self._step_through_video(gaze_idx, "neon")
                acc_neon_opticflow += self._angle_difference_rough(
                    neon_opticflow,
                    self.neon_fov,
                    np.array([self.neon_video.height, self.neon_video.width]),
                )

            if gaze_idx == 0 or (
                self.corresponding_alt_ts_idx[gaze_idx]
                != self.corresponding_alt_ts_idx[gaze_idx - 1]
            ):
                alt_frame, alt_opticflow = self._step_through_video(
                    gaze_idx, "alternative"
                )
                acc_alt_opticflow += self._angle_difference_rough(
                    alt_opticflow,
                    self.alt_fov,
                    np.array([self.alt_video.height, self.alt_video.width]),
                )
            # Neon recording might have some gray frames at the beginning of it.In this case, no feature matching is possible.
            if np.all(neon_frame == 100):
                self.logger.info(f"Neon frame is all gray")
                gaze_alt_camera = gaze_neon.copy()
            else:
                refresh_needed = self._check_if_refresh_needed(
                    gaze_idx,
                    self._angle_difference_rough(
                        last_gaze - gaze_neon,
                        self.neon_fov,
                        np.array([self.neon_video.height, self.neon_video.width]),
                    ),
                    gazes_since_refresh,
                    acc_neon_opticflow,
                    acc_alt_opticflow,
                    refresh_time_thrshld,
                    optic_flow_thrshld,
                    gaze_change_thrshld,
                )
                if refresh_needed:
                    patch_corners = self._get_patch_corners(
                        self.patch_size, gaze_neon, neon_frame.shape
                    )
                    self.correspondences = self.image_matcher.get_correspondences(
                        neon_frame, alt_frame, patch_corners
                    )
                    self.logger.info(
                        f"Matcher was called at {gaze_ts} ({len(self.correspondences['keypoints0'])} correspondences)"
                    )
                    gazes_since_refresh = 0
                    acc_alt_opticflow = 0
                    acc_neon_opticflow = 0
                    last_gaze = gaze_neon.copy()

                filtered_correspondences, new_patch_corners = (
                    self._filter_correspondences(
                        self.correspondences.copy(), gaze_neon, neon_frame.shape
                    )
                )

                gaze_alt_camera = self._transform_point(
                    filtered_correspondences, gaze_neon
                )

            if self._ts_between_video_frames(gaze_ts):
                gaze_alt_camera = self._move_point_to_arbitrary_timestamp(
                    gaze_alt_camera,
                    alt_rel_ts,
                    self.alt_opticflow,
                    gaze_rel_ts - self.alt2neon_offset_sec,
                )

            self.logger.info(f"({gaze_idx}) Gaze mapped to {gaze_alt_camera}")
            self.alt_gaze.loc[
                self.alt_gaze["timestamp [ns]"] == gaze_ts,
                ["gaze x [px]", "gaze y [px]"],
            ] = gaze_alt_camera

            gazes_since_refresh += 1

        saving_path = Path(
            saving_path or self.output_dir / "alternative_camera_gaze.csv"
        )
        saving_path.parent.mkdir(parents=True, exist_ok=True)

        self.alt_gaze.to_csv(saving_path, index=False)
        self.logger.info(f"Gaze mapped to alternative camera saved at {saving_path}")
        return saving_path

    def _create_alternative_gaze_df(self):
        """Creates a DataFrame with the same formatting as the neon_gaze DataFrame, the 'gaze x [px]',
        'gaze y [px]' columns are filled with None values, the 'azimuth [deg]' and 'elevation [deg]' columns
        are dropped, the rest of the columns keep the same values as the neon_gaze DataFrame.

        Returns:
            DataFrame: A DataFrame with similar formatting as the neon_gaze DataFrame
        """
        alt_gaze_dataframe = pd.DataFrame(
            {
                col: [None for _ in self.neon_gaze["timestamp [ns]"].values]
                for col in self.neon_gaze.columns
            }
        )
        alt_gaze_dataframe = alt_gaze_dataframe.drop(
            ["azimuth [deg]", "elevation [deg]"], axis=1
        )
        alt_gaze_dataframe[
            [
                "section id",
                "recording id",
                "timestamp [ns]",
                "worn",
                "fixation id",
                "blink id",
            ]
        ] = self.neon_gaze[
            [
                "section id",
                "recording id",
                "timestamp [ns]",
                "worn",
                "fixation id",
                "blink id",
            ]
        ].values

        alt_gaze_dataframe = alt_gaze_dataframe[
            (
                alt_gaze_dataframe["timestamp [ns]"]
                >= self.alt_vid_ts_nanosec["timestamp [ns]"].values[0]
            )
            & (
                alt_gaze_dataframe["timestamp [ns]"]
                <= self.alt_vid_ts_nanosec["timestamp [ns]"].values[-1]
            )
        ]

        return alt_gaze_dataframe

    def _check_if_refresh_needed(
        self,
        gaze_index,
        angle_between_gazes,
        gazes_since_refresh,
        accumulated_neon_opticflow,
        accumulated_alt_opticflow,
        refresh_thrshld=None,
        opticflow_thrshld=None,
        gaze_change_thrshld=None,
    ):
        if gaze_index == 0:
            return True
        if self.correspondences is None:
            self.logger.info("No correspondences found, refreshing transformation")
            return True

        if (
            gaze_change_thrshld is not None
            and angle_between_gazes > gaze_change_thrshld
        ):
            self.logger.info(
                f"Large gaze jump detected ({angle_between_gazes}deg), refreshing transformation"
            )
            return True

        if (
            opticflow_thrshld is not None
            and accumulated_alt_opticflow > opticflow_thrshld
        ):
            self.logger.info(
                f"Optic flow threshold reached (alternative camera at {np.linalg.norm(accumulated_alt_opticflow)}), refreshing transformation"
            )
            return True

        if (
            opticflow_thrshld is not None
            and accumulated_neon_opticflow > opticflow_thrshld
        ):
            self.logger.info(
                f"Optic flow threshold reached (neon at {np.linalg.norm(accumulated_neon_opticflow)}), refreshing transformation"
            )
            return True

        if refresh_thrshld is not None and gazes_since_refresh >= refresh_thrshld:
            self.logger.info(f"Refreshing transformation after {refresh_thrshld} gazes")
            return True

        if (
            refresh_thrshld is None
            and gaze_change_thrshld is None
            and opticflow_thrshld is None
        ):
            return True

        return False

    def _obtain_relative_ts(self, gaze_ts, gaze_index):
        gaze_relative_ts = (
            gaze_ts - self.neon_vid_ts_nanosec["timestamp [ns]"].values[0]
        ) / 1e9
        neon_relative_ts = self.neon_video.timestamps[
            self.corresponding_neon_ts_idx[gaze_index]
        ]
        action_relative_ts = self.alt_video.timestamps[
            self.corresponding_alt_ts_idx[gaze_index]
        ]
        return gaze_relative_ts, neon_relative_ts, action_relative_ts

    def _step_through_video(self, i, video_type):
        self.logger.info(
            f"{i} Step through video {video_type}(neon={self.corresponding_neon_ts_idx[i]}, alternative_video={self.corresponding_alt_ts_idx[i]})"
        )
        relative_ts = (
            self.neon_video.timestamps[self.corresponding_neon_ts_idx[i]]
            if video_type == "neon"
            else self.alt_video.timestamps[self.corresponding_alt_ts_idx[i]]
        )
        frame = (
            self.neon_video.get_frame_by_timestamp(relative_ts)
            if video_type == "neon"
            else self.alt_video.get_frame_by_timestamp(relative_ts)
        )
        if i > 0:
            opticflow = (
                self.neon_opticflow.loc[
                    self.neon_opticflow["end"] == relative_ts, ["dx", "dy"]
                ].values
                if video_type == "neon"
                else self.alt_opticflow.loc[
                    self.alt_opticflow["end"] == relative_ts, ["dx", "dy"]
                ].values
            )
        else:
            opticflow = np.array([0, 0])

        if self.correspondences is not None:
            selected_kp = "keypoints0" if video_type == "neon" else "keypoints1"
            prev_ts = (
                self.neon_video.timestamps[self.corresponding_neon_ts_idx[i - 1]]
                if video_type == "neon"
                else self.alt_video.timestamps[self.corresponding_alt_ts_idx[i - 1]]
            )
            self.logger.debug(f"Moving {selected_kp} to {relative_ts}")
            self.correspondences[selected_kp] = self._move_point_to_video_timestamp(
                self.correspondences[selected_kp],
                prev_ts,
                relative_ts,
                (self.neon_opticflow if video_type == "neon" else self.alt_opticflow),
            )
        return frame, opticflow

    def _ts_between_video_frames(self, timestamp):
        return (
            max(
                self.neon_vid_ts_nanosec["timestamp [ns]"].values[0],
                self.alt_vid_ts_nanosec["timestamp [ns]"].values[0],
            )
            < timestamp
            < min(
                self.neon_vid_ts_nanosec["timestamp [ns]"].values[-1],
                self.alt_vid_ts_nanosec["timestamp [ns]"].values[-1],
            )
        )

    def _filter_correspondences(
        self, correspondences, point_to_be_transformed, image_shape
    ):
        if len(correspondences["keypoints0"]) < 100:
            self.logger.warning(
                f"Less than 100 correspondences in original patch (size {self.patch_size}), returning {len(correspondences['keypoints0'])} correspondences"
            )
            return correspondences, self._get_patch_corners(
                self.patch_size, point_to_be_transformed, image_shape
            )

        distance_to_point = np.linalg.norm(
            point_to_be_transformed - correspondences["keypoints0"], axis=1
        )

        different_radii = range(50, self.patch_size + 50, 50)
        for radius in different_radii:
            kept_idx = distance_to_point < radius
            if np.count_nonzero(kept_idx) > 100:
                self.logger.info(
                    f"Patch size reduced to {radius*2} with {len(kept_idx)} correspondences"
                )
                for name in correspondences.keys():
                    correspondences[name] = correspondences[name][kept_idx]
                return correspondences, self._get_patch_corners(
                    radius * 2, point_to_be_transformed, image_shape
                )
        self.logger.warning(
            "No radius found with more than 100 correspondences. Returning original correspondences."
        )
        return correspondences, self._get_patch_corners(
            self.patch_size, point_to_be_transformed, image_shape
        )

    def _transform_point(self, correspondences, point_to_be_transformed):
        self._estimate_transformation(correspondences)
        point_to_be_transformed = np.float32(point_to_be_transformed).reshape(-1, 1, 2)
        transformed_point = cv.perspectiveTransform(
            point_to_be_transformed, self.gaze_transformation
        )
        return transformed_point.reshape(1, 2)

    def _estimate_transformation(self, correspondences):
        neon_pts = np.float32(correspondences["keypoints0"]).reshape(-1, 1, 2)
        alt_cam_pts = np.float32(correspondences["keypoints1"]).reshape(-1, 1, 2)
        prev_transformation = self.gaze_transformation
        try:
            self.gaze_transformation, mask = cv.findHomography(
                neon_pts, alt_cam_pts, cv.RANSAC, 5.0
            )
            if mask.ravel().sum() == 0:
                self.logger.error("No inliers found, using previous transformation")
                self.gaze_transformation = prev_transformation
        except cv.error:
            self.logger.error(
                "Homography could not be estimated, using previous transformation"
            )

    def _move_point_to_video_timestamp(
        self, point_coordinates, point_timestamp, opticflow_timestamp, opticflow
    ):
        """Moves a point in an arbitrary timestamp to the closest video frame along the given optic flow.

        Args:
            point_coordinates (ndarray): 2D coordinates of the point to be moved
            point_timestamp (float): Timestamp of the point to be moved
            opticflow_timestamp (float): Timestamp of the closest frame of the optic flow signal
            opticflow (DataFrame): Optic Flow signal of the video

        Returns:
            ndarray: New coordinates of the point after being moved
        """
        time_difference = opticflow_timestamp - point_timestamp
        if np.abs(time_difference) < 1e-6:
            return point_coordinates
        elif time_difference > 0:
            # The point is in the past, so it needs to be moved to the 'future'
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["end"] == opticflow_timestamp, ["dx", "dy"]
            ].values
            dt = np.diff(
                opticflow.loc[
                    opticflow["end"] == opticflow_timestamp, ["start", "end"]
                ].values
            )
            dx_dy_dt = opticflow_displacement_between_frames / dt
        elif time_difference < 0:
            # The point is in the future, so it needs to be moved to the 'past', against the optic flow between the optic flow timestamp and the next timestamp
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["start"] == opticflow_timestamp, ["dx", "dy"]
            ].values
            dt = np.diff(
                opticflow.loc[
                    opticflow["start"] == opticflow_timestamp, ["start", "end"]
                ].values
            )
            dx_dy_dt = opticflow_displacement_between_frames / dt
        dx_dy = dx_dy_dt * time_difference
        return point_coordinates + dx_dy

    def _move_point_to_arbitrary_timestamp(
        self, point_coordinates, video_timestamp, opticflow, target_timestamp
    ):
        time_difference = target_timestamp - video_timestamp
        if time_difference == 0:
            return point_coordinates
        elif (
            time_difference > 0
        ):  # target timestamp is in the future with respect to the video timestamp
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["start"] == video_timestamp, ["dx", "dy"]
            ].values
            dx_dy_dt = opticflow_displacement_between_frames / np.diff(
                opticflow.loc[
                    opticflow["start"] == video_timestamp, ["start", "end"]
                ].values
            )
        elif (
            time_difference < 0
        ):  # target timestamp is in the past with respect to the video timestamp
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["end"] == video_timestamp, ["dx", "dy"]
            ].values
            dx_dy_dt = opticflow_displacement_between_frames / np.diff(
                opticflow.loc[
                    opticflow["end"] == video_timestamp, ["start", "end"]
                ].values
            )
        dx_dy = dx_dy_dt * time_difference
        return point_coordinates + dx_dy

    @staticmethod
    def _get_patch_corners(patch_size, point, image_shape):
        point = point.reshape(-1)
        if point[0] < patch_size // 2:
            x_min = 0
            x_max = patch_size
        elif point[0] < image_shape[1] - patch_size // 2:
            x_min = point[0] - patch_size // 2
            x_max = point[0] + patch_size // 2
        else:
            x_min = image_shape[1] - patch_size
            x_max = image_shape[1]
        if point[1] < patch_size / 2:
            y_min = 0
            y_max = patch_size
        elif point[1] < image_shape[0] - patch_size // 2:
            y_min = point[1] - patch_size // 2
            y_max = point[1] + patch_size // 2
        else:
            y_min = image_shape[0] - patch_size
            y_max = image_shape[0]
        return np.array(
            [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]],
            dtype=np.int32,
        )

    @staticmethod
    def _get_corresponding_timestamps_index(timestamps_1, timestamps_2):
        """For each timestamp in timestamps_2, finds the index of the closest timestamp in timestamps_1
        Args:
            timestamps_1 (ndarray): 1-DIM array with timestamps
            timestamps_2 (ndarray): 1-DIM array with timestamps to be compared with timestamps_1

        Returns:
            ndarray: 1-DIM array with the indexes of the timestamps_1 that are closest to the timestamps_2. Same length as timestamps_2
        """
        after_idxs = np.searchsorted(timestamps_1, timestamps_2)
        before_idxs = after_idxs - 1
        before_idxs[before_idxs == -1] = 0
        after_idxs[after_idxs == len(timestamps_1)] = len(timestamps_1) - 1

        after_diff = timestamps_1[after_idxs] - timestamps_2
        before_diff = timestamps_2 - timestamps_1[before_idxs]

        stacked_diff = np.array([after_diff, before_diff])
        stacked_idxs = np.array([after_idxs, before_idxs])
        selected_indexes = np.argmin(np.abs(stacked_diff), axis=0, keepdims=True)
        indexes = np.take_along_axis(stacked_idxs, selected_indexes, axis=0)
        indexes.shape = -1
        return indexes

    @staticmethod
    def _angle_difference_rough(pixel_displacement, fov, image_resolution):
        """Calculates a rough estimate of the angle difference between two points in the same plane

        Args:
            pixel_displacement (ndarray): Displacement in x and y in pixels
            fov (ndarray): Field of view of the camera (can be 2D or 1D). If 1D, it is assumed that the camera has the same field of view in both axes. If 2D, it is assumed that [fov_x, fov_y]
            image_resolution (ndarray): Resolution of the image (can be 2D or 1D). If 1D, it is assumed that the image has the same resolution in both axes. If 2D, it is assumed that [height, width]
        """
        if len(fov) == 1:
            fov = np.array([fov, fov])
        fov = fov.reshape(-1)
        image_resolution = image_resolution.reshape(-1)

        angle_per_pixel_x = fov[0] / image_resolution[1]
        angle_per_pixel_y = fov[1] / image_resolution[0]

        angle_difference = pixel_displacement.reshape(-1) * np.array(
            [angle_per_pixel_x, angle_per_pixel_y]
        )
        return np.linalg.norm(angle_difference)

    @staticmethod
    def _angular_difference(point_a, point_b, camera_matrix, distortion_coefficients):
        """Using camera instrinsics, calculates the angular difference between two points in the same plane"""
        point_a = cv.undistortPoints(point_a, camera_matrix, distortion_coefficients)
        point_b = cv.undistortPoints(point_b, camera_matrix, distortion_coefficients)
        point_a = np.concatenate([point_a.reshape(-1), np.ones(1)])
        point_b = np.concatenate([point_b.reshape(-1), np.ones(1)])
        angular_difference = np.arccos(
            point_a.dot(point_b) / (np.linalg.norm(point_a) * np.linalg.norm(point_b))
        )
        return np.rad2deg(angular_difference)

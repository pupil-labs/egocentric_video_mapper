from tqdm import tqdm
import cv2 as cv
import logging
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
        alternative_video_dir,
        neon_timestamps,
        alternative_timestamps,
        image_matcher,
        image_matcher_parameters,
        neon_opticflow_csv=None,
        alternative_opticflow_csv=None,
        patch_size=1000,
        verbose=True,
    ) -> None:
        self.neon_video = VideoHandler(neon_video_dir)  # name consistency
        self.alternative_video = VideoHandler(alternative_video_dir)

        self.neon_ts = pd.read_csv(neon_timestamps)
        self.alternative_ts = pd.read_csv(alternative_timestamps)
        self.alt2neon_offset = (
            self.alternative_ts["timestamp [ns]"].values[0]
            - self.neon_ts["timestamp [ns]"].values[0]
        ) / 1e9

        self.neon_opticflow = pd.read_csv(neon_opticflow_csv, dtype=np.float32)
        self.alternative_opticflow = pd.read_csv(
            alternative_opticflow_csv, dtype=np.float32
        )

        self.neon_gaze = pd.read_csv(neon_gaze_csv)  # <- at 200Hz
        self.alternative_gaze = self._create_alternative_gaze_df()

        self.image_matcher = ImageMatcherFactory().get_matcher(
            image_matcher, image_matcher_parameters
        )

        self.patch_size = patch_size
        self.transformation = np.array(
            [
                [self.alternative_video.width / self.neon_video.width, 0, 0],
                [0, self.alternative_video.height / self.neon_video.height, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.verbose = verbose

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

    def map_gaze(self, saving_path=None):

        for i, gaze_ts in enumerate(
            tqdm(
                self.alternative_gaze["timestamp [ns]"].values,
                desc="Mapping gaze signal",
                total=len(self.alternative_gaze["timestamp [ns]"].values),
            )
        ):
            gaze_neon = self.neon_gaze.loc[
                self.neon_gaze["timestamp [ns]"] == gaze_ts,
                ["gaze x [px]", "gaze y [px]"],
            ].values.reshape(1, 2)

            gaze_relative_ts = (
                gaze_ts - self.neon_ts["timestamp [ns]"].values[0]
            ) / 1e9
            neon_relative_ts = self.neon_video.get_closest_timestamp(gaze_relative_ts)[
                0
            ]
            alt_relative_ts = self.alternative_video.get_closest_timestamp(
                neon_relative_ts - self.alt2neon_offset
            )[0]
            self.logger.info(f"({i}) Calculating gaze {gaze_neon} at {gaze_ts}")

            if self._gaze_between_video_frames(gaze_ts):
                semantically_correct_gaze = self._move_point_to_video_timestamp(
                    gaze_neon,
                    gaze_relative_ts,
                    neon_relative_ts,
                    self.neon_opticflow,
                )
                gaze_alternative_camera = self._map_one_gaze(
                    semantically_correct_gaze, neon_relative_ts, alt_relative_ts
                )
                gaze_alternative_camera = self._move_point_to_arbitrary_timestamp(
                    gaze_alternative_camera,
                    alt_relative_ts,
                    self.alternative_opticflow,
                    gaze_relative_ts - self.alt2neon_offset,
                )
            else:
                gaze_alternative_camera = self._map_one_gaze(
                    gaze_neon, neon_relative_ts, alt_relative_ts
                )

            self.logger.info(
                f"({i}) Gaze ({gaze_neon}) at {gaze_ts} mapped to {gaze_alternative_camera}"
            )

            self.alternative_gaze.loc[
                self.alternative_gaze["timestamp [ns]"] == gaze_ts,
                ["gaze x [px]", "gaze y [px]"],
            ] = gaze_alternative_camera

        if saving_path is None:
            saving_path = (
                Path(self.neon_video.path).parent / "alternative_camera_gaze.csv"
            )

        else:
            Path(saving_path).parent.mkdir(parents=True, exist_ok=True)
        self.alternative_gaze.to_csv(saving_path, index=False)
        return saving_path

    def _create_alternative_gaze_df(self):
        """Creates a DataFrame with the same formatting as the neon_gaze DataFrame, the 'gaze x [px]',
        'gaze y [px]', 'azimuth [deg]' and 'elevation [deg]' columns are filled with None values, while the rest of the columns keep the same values as the neon_gaze DataFrame.

        Returns:
            DataFrame: A DataFrame with the same formatting as the neon_gaze DataFrame
        """
        alternative_dataframe = pd.DataFrame.from_dict(
            {
                col: [None for _ in self.neon_gaze["timestamp [ns]"].values]
                for col in self.neon_gaze.columns
            }
        )

        alternative_dataframe.loc[
            :,
            [
                "section id",
                "recording id",
                "timestamp [ns]",
                "worn",
                "fixation id",
                "blink id",
            ],
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

        alternative_dataframe = alternative_dataframe[
            alternative_dataframe["timestamp [ns]"]
            >= self.alternative_ts["timestamp [ns]"].values[0]
        ]
        alternative_dataframe = alternative_dataframe[
            alternative_dataframe["timestamp [ns]"]
            <= self.alternative_ts["timestamp [ns]"].values[-1]
        ]

        return alternative_dataframe

    def _gaze_between_video_frames(self, gaze_timestamp):
        return (
            max(
                self.neon_ts["timestamp [ns]"].values[0],
                self.alternative_ts["timestamp [ns]"].values[0],
            )
            < gaze_timestamp
            < min(
                self.neon_ts["timestamp [ns]"].values[-1],
                self.alternative_ts["timestamp [ns]"].values[-1],
            )
        )

    def _map_one_gaze(self, gaze_coords, neon_timestamp, alt_timestamp):
        alternative_frame = self.alternative_video.get_frame_by_timestamp(alt_timestamp)
        neon_frame = self.neon_video.get_frame_by_timestamp(neon_timestamp)

        if np.all(neon_frame == 100):
            self.logger.warning(f"Neon frame at {neon_timestamp} is all gray")
            return gaze_coords

        patch_corners = self._get_patch_corners(
            self.patch_size, gaze_coords, neon_frame.shape
        )
        correspondences = self.image_matcher.get_correspondences(
            neon_frame, alternative_frame, patch_corners
        )
        correspondences, new_patch_corners = self._filter_correspondences(
            correspondences.copy(), gaze_coords, neon_frame.shape
        )
        self.logger.info(
            f'Number of correspondences: {len(correspondences["keypoints0"])} at {abs(new_patch_corners[0,0]-new_patch_corners[2,0])} patch size'
        )
        gaze_in_alternative_camera = self._transform_point(correspondences, gaze_coords)
        return gaze_in_alternative_camera

    def _transform_point(self, correspondences, point_to_be_transformed):
        self._estimate_transformation(correspondences)
        point_to_be_transformed = np.float32(point_to_be_transformed).reshape(-1, 1, 2)
        transformed_point = cv.perspectiveTransform(
            point_to_be_transformed, self.transformation
        )
        return transformed_point.reshape(1, 2)

    def _estimate_transformation(self, correspondences):
        # returns callable

        neon_pts = np.float32(correspondences["keypoints0"]).reshape(-1, 1, 2)
        alt_cam_pts = np.float32(correspondences["keypoints1"]).reshape(-1, 1, 2)
        prev_transformation = self.transformation
        try:
            self.transformation, mask = cv.findHomography(
                neon_pts, alt_cam_pts, cv.RANSAC, 5.0
            )
            if mask.ravel().sum() == 0:
                self.logger.error("No inliers found, using previous transformation")
                # may be better to not do this, rather just not map the gaze and leave it empty
                self.transformation = prev_transformation
        except cv.error:
            self.logger.error(
                "Homography could not be estimated, using previous transformation"
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

        diferent_radii = range(50, self.patch_size + 50, 50)
        for radius in diferent_radii:
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
        if time_difference == 0:
            return point_coordinates
        elif (
            time_difference > 0
        ):  # The point is in the past, so it needs to be moved to the 'future'
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["end"] == opticflow_timestamp, ["dx", "dy"]
            ].values
            dx_dy_dt = opticflow_displacement_between_frames / np.diff(
                opticflow.loc[
                    opticflow["end"] == opticflow_timestamp, ["start", "end"]
                ].values
            )
        elif (
            time_difference < 0
        ):  # The point is in the future, so it needs to be moved to the 'past', against the optic flow between the optic flow timestamp and the next timestamp
            opticflow_displacement_between_frames = opticflow.loc[
                opticflow["start"] == opticflow_timestamp, ["dx", "dy"]
            ].values
            dx_dy_dt = opticflow_displacement_between_frames / np.diff(
                opticflow.loc[
                    opticflow["start"] == opticflow_timestamp, ["start", "end"]
                ].values
            )
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


class ActionCameraGazeMapper2(ActionCameraGazeMapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corresponding_altcam_idx = self._get_corresponding_timestamps_index(
            self.alternative_ts["timestamp [ns]"].values,
            self.neon_ts["timestamp [ns]"].values,
        )

    def map_gaze(self, saving_path=None):
        neon_start_idx = self.neon_video.get_closest_timestamp(
            (
                self.alternative_gaze["timestamp [ns]"].values[0]
                - self.neon_ts["timestamp [ns]"].values[0]
            )
            / 1e9
        )[1]
        for i, neon_ts in enumerate(
            self.neon_ts["timestamp [ns]"].values[neon_start_idx:],
            start=neon_start_idx,
        ):
            lower_bound = (
                neon_ts - (neon_ts - self.neon_ts["timestamp [ns]"].values[i - 1]) / 2
                if i > 0
                else 0
            )
            upper_bound = (
                neon_ts + (self.neon_ts["timestamp [ns]"].values[i + 1] - neon_ts) / 2
                if i < len(self.neon_ts["timestamp [ns]"].values) - 1
                else np.inf
            )
            gazes_for_this_frame = self.alternative_gaze.loc[
                self.alternative_gaze["timestamp [ns]"].between(
                    lower_bound, upper_bound, inclusive="left"
                )
            ]

            neon_relative_ts = self.neon_video.timestamps[i]
            action_relative_ts = self.alternative_video.timestamps[
                self.corresponding_altcam_idx[i]
            ]
            self.logger.info(
                f'({i}) Mapping {len(gazes_for_this_frame)} gazes from neon frame at {neon_ts}({neon_relative_ts}) to action camera frame at {self.alternative_ts["timestamp [ns]"].values[self.corresponding_altcam_idx[i]]}({action_relative_ts})'
            )
            print(
                f'({i}) Mapping {len(gazes_for_this_frame)} gazes from neon frame at {neon_ts}({neon_relative_ts}) to action camera frame at {self.alternative_ts["timestamp [ns]"].values[self.corresponding_altcam_idx[i]]}({action_relative_ts})'
            )
            neon_frame = self.neon_video.get_frame_by_timestamp(neon_relative_ts)
            action_frame = self.alternative_video.get_frame_by_timestamp(
                action_relative_ts
            )

            for gaze_ts in gazes_for_this_frame["timestamp [ns]"].values:
                gaze_neon = self.neon_gaze.loc[
                    self.neon_gaze["timestamp [ns]"] == gaze_ts,
                    ["gaze x [px]", "gaze y [px]"],
                ].values.reshape(1, 2)
                gaze_relative_ts = (
                    gaze_ts - self.neon_ts["timestamp [ns]"].values[0]
                ) / 1e9
                if self._gaze_between_video_frames(gaze_ts):
                    semantic_gaze = self._move_point_to_video_timestamp(
                        gaze_neon,
                        gaze_relative_ts,
                        neon_relative_ts,
                        self.neon_opticflow,
                    )
                    gaze_action_camera = self._map_one_gaze(
                        semantic_gaze, neon_frame, action_frame
                    )[0]
                    print(f"AC Gaze before moving it:{gaze_action_camera}")
                    gaze_action_camera = self._move_point_to_arbitrary_timestamp(
                        gaze_action_camera,
                        action_relative_ts,
                        self.alternative_opticflow,
                        gaze_relative_ts - self.alt2neon_offset,
                    )
                else:
                    gaze_action_camera = self._map_one_gaze(
                        gaze_neon, neon_frame, action_frame
                    )[0]
                self.logger.info(
                    f"- Gaze ({gaze_neon}) at {gaze_relative_ts} mapped to {gaze_action_camera}"
                )
                print(
                    f"- Gaze ({gaze_neon}) at {gaze_relative_ts} mapped to {gaze_action_camera}"
                )
                self.alternative_gaze.loc[
                    self.alternative_gaze["timestamp [ns]"] == gaze_ts,
                    ["gaze x [px]", "gaze y [px]"],
                ] = gaze_action_camera

        if saving_path is None:
            self.alternative_gaze.to_csv(
                Path(self.neon_video.path).parent / "action_gaze.csv", index=False
            )
        else:
            self.alternative_gaze.to_csv(saving_path, index=False)

    def _map_one_gaze(self, gaze_coordinates, neon_frame, action_frame):
        if np.all(neon_frame == 100):
            self.logger.warning(f"Neon frame is all gray")
            return gaze_coordinates
        patch_corners = self._get_patch_corners(
            self.patch_size, gaze_coordinates, neon_frame.shape
        )
        correspondences = self.image_matcher.get_correspondences(
            neon_frame, action_frame, patch_corners
        )
        filt_correspondences, new_patch_corners = self._filter_correspondences(
            correspondences.copy(), gaze_coordinates, neon_frame.shape
        )
        self.logger.info(
            f'Number of correspondences: {len(filt_correspondences["keypoints0"])} at {abs(new_patch_corners[0,0]-new_patch_corners[2,0])} patch size'
        )
        print(
            f'Number of correspondences: {len(filt_correspondences["keypoints0"])} at {abs(new_patch_corners[0,0]-new_patch_corners[2,0])} patch size'
        )
        gaze_in_action_camera = self._transform_point(
            filt_correspondences, gaze_coordinates
        )
        return gaze_in_action_camera, correspondences

    @staticmethod
    def _get_corresponding_timestamps_index(timestamps_1, timestamps_2):
        """For each timestamp in timestamps_2, finds the index of the closest timestamp in timestamps_1
        Args:
            timestamps_1 (_type_): 1-DIM array with timestamps
            timestamps_2 (_type_): 1-DIM array with timestamps to be compared with timestamps_1

        Returns:
            ndarray: 1-DIM array with the indexes of the timestamps_1 that are closest to the timestamps_2. Same length as timestamps_2
        """
        after_posiciones = np.searchsorted(timestamps_1, timestamps_2)
        before_posiciones = after_posiciones - 1
        before_posiciones[before_posiciones == -1] = 0
        after_posiciones[after_posiciones == len(timestamps_1)] = len(timestamps_1) - 1

        after_diff = timestamps_1[after_posiciones] - timestamps_2
        before_diff = timestamps_2 - timestamps_1[before_posiciones]

        stacked_diff = np.array([after_diff, before_diff])
        stacked_posiciones = np.array([after_posiciones, before_posiciones])
        selected_indexes = np.argmin(np.abs(stacked_diff), axis=0, keepdims=True)
        indexes = np.take_along_axis(stacked_posiciones, selected_indexes, axis=0)
        indexes.shape = -1
        return indexes


class RulesBasedGazeMapper(ActionCameraGazeMapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corresponding_alternative_idx = self._get_corresponding_timestamps_index(
            self.alternative_ts["timestamp [ns]"].values,
            self.alternative_gaze["timestamp [ns]"].values,
        )
        self.corresponding_neon_idx = self._get_corresponding_timestamps_index(
            self.neon_ts["timestamp [ns]"].values,
            self.alternative_gaze["timestamp [ns]"].values,
        )
        self.neon_fov = np.array([103, 81])
        self.alternative_fov = np.array([145, 76])

    def map_gaze(
        self,
        saving_path=None,
        refresh_time_thrshld=None,
        optic_flow_thrshld=None,
        gaze_change_thrshld=None,
    ):
        """

        Args:
            saving_path (str, optional): Saving path for the mapped gaze, in format /path/to/action_gaze.csv, if None is given it saves the mapped gaze in the parent directory of the neon video. Defaults to None.
            refresh_time_thrshld (int, optional): Maximum allowed time elapsed, in seconds, since the last computation of image correspondences. If set to None this threshold is not enforced. Defaults to None.
            opticf_threshold (int, optional): Maximum allowed cummulative optic flow, in pixels, since the last computation of image correspondences. If set to None this threshold is not enforced Defaults to None.
            gaze_change_thrshld (int, optional): Maximum allowed neon gaze change, in  pixels, since the last computation of image correspondences. If set to None this threshold is not enforced. Defaults to None.
        """
        refresh_time_thrshld = (
            refresh_time_thrshld * 200 if refresh_time_thrshld is not None else None
        )

        gazes_since_refresh = 0
        acc_alternative_opticflow = 0
        acc_neon_opticflow = 0
        last_gaze = self.neon_gaze.iloc[0][["gaze x [px]", "gaze y [px]"]].values

        for i, gaze_ts in enumerate(
            tqdm(
                self.alternative_gaze["timestamp [ns]"].values,
                desc="Mapping gaze signal",
                total=len(self.alternative_gaze["timestamp [ns]"].values),
            )
        ):

            gaze_neon = self.neon_gaze.loc[
                self.neon_gaze["timestamp [ns]"] == gaze_ts,
                ["gaze x [px]", "gaze y [px]"],
            ].values.reshape(1, 2)

            gaze_rel_ts, neon_rel_ts, alt_rel_ts = self._obtain_relative_ts(gaze_ts, i)
            self.logger.info(f"({i}) Transforming gaze {gaze_neon} at {gaze_ts}")

            if self._gaze_between_video_frames(gaze_ts):
                gaze_neon = self._move_point_to_video_timestamp(
                    gaze_neon,
                    gaze_rel_ts,
                    neon_rel_ts,
                    self.neon_opticflow,
                )

            # check if new frames need to be retrieved
            if i == 0 or (
                self.corresponding_neon_idx[i] != self.corresponding_neon_idx[i - 1]
            ):
                neon_frame, neon_opticflow = self._step_through_video(i, "neon")
                acc_neon_opticflow += self._angle_difference_rough(
                    neon_opticflow,
                    self.neon_fov,
                    np.array([self.neon_video.height, self.neon_video.width]),
                )

            if i == 0 or (
                self.corresponding_alternative_idx[i]
                != self.corresponding_alternative_idx[i - 1]
            ):
                alternative_frame, alternative_opticflow = self._step_through_video(
                    i, "alternative"
                )
                acc_alternative_opticflow += self._angle_difference_rough(
                    alternative_opticflow,
                    self.alternative_fov,
                    np.array(
                        [self.alternative_video.height, self.alternative_video.width]
                    ),
                )

            # check if neon frame is all gray
            if np.all(neon_frame == 100):
                self.logger.warning(f"Neon frame is all gray")
                gaze_alternative_camera = gaze_neon.copy()

            else:
                # check if matcher needs to be called
                if i == 0 or self._check_if_refresh_needed(
                    self._angle_difference_rough(
                        last_gaze - gaze_neon,
                        self.neon_fov,
                        np.array([self.neon_video.height, self.neon_video.width]),
                    ),
                    gazes_since_refresh,
                    acc_neon_opticflow,
                    acc_alternative_opticflow,
                    refresh_time_thrshld,
                    optic_flow_thrshld,
                    gaze_change_thrshld,
                ):
                    patch_corners = self._get_patch_corners(
                        self.patch_size, gaze_neon, neon_frame.shape
                    )
                    self.correspondences = self.image_matcher.get_correspondences(
                        neon_frame, alternative_frame, patch_corners
                    )
                    self.logger.warning(
                        f"Matcher was called at {gaze_ts} ({len(self.correspondences['keypoints0'])} correspondences)"
                    )
                    gazes_since_refresh = 0
                    acc_alternative_opticflow = 0
                    acc_neon_opticflow = 0
                    last_gaze = gaze_neon.copy()

                filt_correspondences, new_patch_corners = self._filter_correspondences(
                    self.correspondences.copy(), gaze_neon, neon_frame.shape
                )

                gaze_alternative_camera = self._transform_point(
                    filt_correspondences, gaze_neon
                )

            if self._gaze_between_video_frames(gaze_ts):
                gaze_alternative_camera = self._move_point_to_arbitrary_timestamp(
                    gaze_alternative_camera,
                    alt_rel_ts,
                    self.alternative_opticflow,
                    gaze_rel_ts - self.alt2neon_offset,
                )

            self.logger.info(f"({i}) Gaze mapped to {gaze_alternative_camera}")
            self.alternative_gaze.loc[
                self.alternative_gaze["timestamp [ns]"] == gaze_ts,
                ["gaze x [px]", "gaze y [px]"],
            ] = gaze_alternative_camera

            gazes_since_refresh += 1

        if saving_path is None:
            saving_path = (
                Path(self.neon_video.path).parent / "alternative_camera_gaze.csv"
            )
        else:
            Path(saving_path).parent.mkdir(parents=True, exist_ok=True)

        self.alternative_gaze.to_csv(
            saving_path,
            index=False,
        )
        print(f"Gaze mapped to alternative camera saved at {saving_path}")
        return saving_path

    def _obtain_relative_ts(self, gaze_ts, gaze_index):
        gaze_relative_ts = (gaze_ts - self.neon_ts["timestamp [ns]"].values[0]) / 1e9
        neon_relative_ts = self.neon_video.timestamps[
            self.corresponding_neon_idx[gaze_index]
        ]
        action_relative_ts = self.alternative_video.timestamps[
            self.corresponding_alternative_idx[gaze_index]
        ]
        return gaze_relative_ts, neon_relative_ts, action_relative_ts

    def _step_through_video(self, i, video_type):
        if self.verbose:
            print(
                f"{i} Step through video {video_type}(neon={self.corresponding_neon_idx[i]}, alternative_video={self.corresponding_alternative_idx[i]})"
            )
        relative_ts = (
            self.neon_video.timestamps[self.corresponding_neon_idx[i]]
            if video_type == "neon"
            else self.alternative_video.timestamps[
                self.corresponding_alternative_idx[i]
            ]
        )
        frame = (
            self.neon_video.get_frame_by_timestamp(relative_ts)
            if video_type == "neon"
            else self.alternative_video.get_frame_by_timestamp(relative_ts)
        )
        if i > 0:
            opticflow = (
                self.neon_opticflow.loc[
                    self.neon_opticflow["end"] == relative_ts, ["dx", "dy"]
                ].values
                if video_type == "neon"
                else self.alternative_opticflow.loc[
                    self.alternative_opticflow["end"] == relative_ts, ["dx", "dy"]
                ].values
            )
        else:
            opticflow = np.array([0, 0])

        if hasattr(self, "correspondences"):
            selected_kp = "keypoints0" if video_type == "neon" else "keypoints1"
            prev_ts = (
                self.neon_video.timestamps[self.corresponding_neon_idx[i - 1]]
                if video_type == "neon"
                else self.alternative_video.timestamps[
                    self.corresponding_alternative_idx[i - 1]
                ]
            )
            if self.verbose:
                print(f"Moving {selected_kp} to {relative_ts}")
            self.correspondences[selected_kp] = self._move_point_to_video_timestamp(
                self.correspondences[selected_kp],
                prev_ts,
                relative_ts,
                (
                    self.neon_opticflow
                    if video_type == "neon"
                    else self.alternative_opticflow
                ),
            )
        return frame, opticflow

    def _check_if_refresh_needed(
        self,
        angle_between_gazes,
        gazes_since_refresh,
        accumulated_neon_opticflow,
        accumulated_alt_opticflow,
        refresh_thrshld=None,
        opticf_threshold=None,
        gaze_change_thrshld=None,
    ):
        refresh_needed = False

        if not hasattr(self, "correspondences"):
            self.logger.info("No correspondences found, refreshing transformation")
            if self.verbose:
                print("No correspondences found, refreshing transformation")
            refresh_needed = True

        if (
            gaze_change_thrshld is not None
            and angle_between_gazes > gaze_change_thrshld
        ):
            self.logger.info(
                f"Large gaze jump detected ({angle_between_gazes}deg), refreshing transformation"
            )
            if self.verbose:
                print(
                    f"Large gaze jump detected ({angle_between_gazes}deg), refreshing transformation"
                )
            refresh_needed = True

        if (
            opticf_threshold is not None
            and accumulated_alt_opticflow > opticf_threshold
        ):
            self.logger.info(
                f"Optic flow threshold reached (alternative camera at {np.linalg.norm(accumulated_alt_opticflow)}), refreshing transformation"
            )
            if self.verbose:
                print(
                    f"Optic flow threshold reached (alternative camera at {np.linalg.norm(accumulated_alt_opticflow)}), refreshing transformation"
                )
            refresh_needed = True

        if (
            opticf_threshold is not None
            and accumulated_neon_opticflow > opticf_threshold
        ):
            self.logger.info(
                f"Optic flow threshold reached (neon at {np.linalg.norm(accumulated_neon_opticflow)}), refreshing transformation"
            )
            if self.verbose:
                print(
                    f"Optic flow threshold reached (neon at {np.linalg.norm(accumulated_neon_opticflow)}), refreshing transformation"
                )
            refresh_needed = True

        if refresh_thrshld is not None and gazes_since_refresh == refresh_thrshld:
            if self.verbose:
                print(f"Refreshing transformation after {refresh_thrshld} gazes")
            self.logger.info(f"Refreshing transformation after {refresh_thrshld} gazes")
            refresh_needed = True

        if (
            refresh_thrshld is None
            and gaze_change_thrshld is None
            and opticf_threshold is None
        ):
            refresh_needed = True

        return refresh_needed

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
        """Calculates the angle difference between two points in the same plane [rough calculation]

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

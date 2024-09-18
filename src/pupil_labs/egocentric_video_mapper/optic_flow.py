import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from pupil_labs.egocentric_video_mapper.video_handler import VideoHandler

logger = logging.getLogger(__name__)


@dataclass
class OpticFlowResult:
    dx: float
    dy: float
    start: float = None
    end: float = None
    angle: float = field(init=False)

    def __post_init__(self):
        self.angle = np.arctan2(self.dy, self.dx)


class OpticFlowCalculatorBase(ABC):
    def __init__(self, video_path):
        self.video_handler = VideoHandler(video_path)
        self.results = pd.DataFrame.from_dict(
            dict(start=[], end=[], dx=[], dy=[], angle=[])
        )

    def process_video(
        self, start_time_sec=None, end_time_sec=None, output_file_path=None
    ):
        """Method to calculate the optic flow in a defined video interval. Optic flow is calculated between consecutive frames in the interval.
        Args:
            start_time_sec (float): start time in seconds of the video interval to calculate the optic flow signal. Defaults to None. If not specified, the start time is the first timestamp in the video.
            end_time_sec (float): end time in seconds of the video interval to calculate the optic flow signal. Defaults to None. If not specified, the end time is the last timestamp in the video.
            output_file_path (str, optional): Path to a csv file to store the calculated optic flow signal. Defaults to None. If not specified, the signal is not saved to a file.

        Returns:
            DataFrame: Pandas DataFrame containing the calculated optic flow signal. Has the following columns: 'start', 'end', 'dx', 'dy', 'angle'.
        """
        if start_time_sec is None:
            start_time_sec = self.video_handler.timestamps[0]
        if end_time_sec is None:
            end_time_sec = self.video_handler.timestamps[-1]
        selected_timestamps = self.video_handler.get_timestamps_in_interval(
            start_time_sec, end_time_sec
        )

        requested_optic_flow = dict(start=[], end=[], dx=[], dy=[], angle=[])
        for ts1, ts2 in tqdm(
            zip(selected_timestamps[:-1], selected_timestamps[1:]),
            desc="Calculating optic flow",
            total=len(selected_timestamps) - 1,
        ):
            if self._is_optic_flow_already_calculated(ts1, ts2):
                flow = self._retrieve_optic_flow(ts1, ts2)
            else:
                frame1 = self.video_handler.get_frame_by_timestamp(ts1)
                frame2 = self.video_handler.get_frame_by_timestamp(ts2)
                flow = self._calculate_optic_flow_between_frames(
                    frame1, frame2, ts1, ts2
                )
            for key, value in asdict(flow).items():
                requested_optic_flow[key].append(value)
        requested_optic_flow = pd.DataFrame.from_dict(
            requested_optic_flow, orient="columns"
        )
        self.results = pd.concat(
            [self.results if not self.results.empty else None, requested_optic_flow],
            ignore_index=True,
        )
        self.results.drop_duplicates(subset=["start", "end"], keep="last", inplace=True)
        self.results.sort_values(by=["start"], inplace=True, ignore_index=True)
        if output_file_path:
            self.write_to_csv(output_file_path, requested_optic_flow)
        return requested_optic_flow

    def _is_optic_flow_already_calculated(self, start_ts, end_ts):
        return not self.results[
            (self.results["start"] == start_ts) & (self.results["end"] == end_ts)
        ].empty

    def _retrieve_optic_flow(self, start_ts, end_ts):
        optic_flow_query = self.results[
            (self.results["start"] == start_ts) & (self.results["end"] == end_ts)
        ].drop(columns=["angle"])
        return (
            OpticFlowResult(**optic_flow_query.to_dict(orient="records")[0])
            if not optic_flow_query.empty
            else None
        )

    @abstractmethod
    def _calculate_optic_flow_between_frames(
        self, first_frame, second_frame, first_ts=None, second_ts=None
    ) -> OpticFlowResult:
        """Method to calculate the optic flow between two frames. This method should be implemented by the child class.

        Args:
            first_frame (ndarray): Image data of the first frame, this frame  is the frame at first_ts.
            second_frame (ndarray): Image data of the second frame, this frame is the frame at second_ts.
            first_ts (float, optional): Timestamp in seconds of the first frame. Defaults to None.
            second_ts (float, optional): Timestamp in seconds of the second frame. Defaults to None.

        Returns:
            OpticFlowResult: Object containing the optic flow data between the two frames.
        """
        return

    def write_to_csv(self, output_file_path, optic_flow_data=None):
        """Method to write the optic flow data to a csv file. If the file already exists, the data is appended to it, duplicates are removed and the file is sorted by the 'start' column to ensure the data is in chronological order.
        Args:
            output_file_path (str): Path to a csv file to store the calculated optic flow signal.
            optic_flow_data (DataFrame, optional): Pandas DataFrame containing an optic flow signal. Defaults to None. If not specified, the optic flow signal stored in results attribute is used.
        """
        if optic_flow_data is None:
            optic_flow_data = self.results
        if Path(output_file_path).exists():
            logger.warning("File already exists, appending to it")
            optic_flow_file = pd.read_csv(output_file_path, dtype=np.float32)
            optic_flow_data = pd.concat(
                [optic_flow_file, optic_flow_data], ignore_index=True
            )
        else:
            Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        optic_flow_data.drop_duplicates(
            subset=["start", "end"], keep="last", inplace=True
        )
        optic_flow_data.sort_values(by=["start"], inplace=True, ignore_index=True)
        # Add a duplicate of the first row in the index 0 to make signal same length as timestamps
        optic_flow_data = pd.concat(
            [optic_flow_data.iloc[[0]], optic_flow_data], ignore_index=True
        )
        optic_flow_data.to_csv(output_file_path, index=False)
        logger.info(f"Optic flow data saved to {output_file_path}")


class OpticFlowCalculatorLK(OpticFlowCalculatorBase):
    """Class to calculate optic flow using the Lucas-Kanade method. Backward and forward flow are calculated and compared to ensure the accuracy of points found in the forward flow.

    Args:
        video_path (str): Path to the video file.
        grid_spacing (int, optional): Spacing between grid points to track. Defaults to 50. The smaller the spacing, the more points are tracked and the more time expensive the calculation is.
        params (dict, optional): Parameters for the Lucas-Kanade method, check OpenCV documentation. Defaults to {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)}.
    """

    def __init__(
        self,
        video_path,
        grid_spacing=50,
        params=None,
    ):
        super().__init__(video_path)
        self.grid_spacing = grid_spacing
        self.grid_points = self._create_grid_points()
        if params is None:
            self.lk_params = {
                "winSize": (15, 15),
                "maxLevel": 2,
                "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            }
        else:
            self.lk_params = params

    def _create_grid_points(self):
        xx, yy = np.mgrid[
            0 : self.video_handler.width : self.grid_spacing,
            0 : self.video_handler.height : self.grid_spacing,
        ]
        point_coordinates = np.array(
            [
                [xx[i, j], yy[i, j]]
                for i in range(xx.shape[0])
                for j in range(xx.shape[1])
            ]
        )
        return point_coordinates.reshape(-1, 1, 2).astype(np.float32)

    def _calculate_optic_flow_between_frames(
        self, first_frame, second_frame, first_ts=None, second_ts=None
    ):
        first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        second_frame = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)

        pts_second_frame, _, _ = cv.calcOpticalFlowPyrLK(
            first_frame, second_frame, self.grid_points, None, **self.lk_params
        )
        backtrack_pts_first_frame, _, _ = cv.calcOpticalFlowPyrLK(
            second_frame, first_frame, pts_second_frame, None, **self.lk_params
        )

        chebyshev_distance = (
            abs(self.grid_points - backtrack_pts_first_frame).reshape(-1, 2).max(-1)
        )
        status = chebyshev_distance < 1.0
        good_new_points, good_old_points = (
            pts_second_frame[status],
            self.grid_points[status],
        )

        displacement = good_new_points.reshape(-1, 2) - good_old_points.reshape(-1, 2)

        return OpticFlowResult(
            dx=np.mean(displacement[:, 0]),
            dy=np.mean(displacement[:, 1]),
            start=first_ts,
            end=second_ts,
        )


class OpticFlowCalculatorFarneback(OpticFlowCalculatorBase):
    """Class to calculate dense optic flow using the  Gunnar Farneback's algorithm.

    Args:
        video_path (str): Path to the video file.
        params (dict, optional): Parameters for the Gunnar Farneback algorithm. Defaults to {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0}.
    """

    def __init__(
        self,
        video_path,
        params=None,
    ):
        super().__init__(video_path)
        if params is None:
            self.farneback_params = {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            }
        else:
            self.farneback_params = params

    def _calculate_optic_flow_between_frames(
        self, first_frame, second_frame, first_ts=None, second_ts=None
    ):
        dense_optic_flow = cv.calcOpticalFlowFarneback(
            cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY),
            cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY),
            None,
            **self.farneback_params,
        )

        avg_displacement_x = np.mean(dense_optic_flow[:, :, 0])
        avg_displacement_y = np.mean(dense_optic_flow[:, :, 1])

        return OpticFlowResult(
            dx=avg_displacement_x, dy=avg_displacement_y, start=first_ts, end=second_ts
        )


def calculate_optic_flow(
    neon_timeseries_dir,
    alternative_video_path,
    output_dir,
    optic_flow_method="farneback",
):
    neon_video_path = next(Path(neon_timeseries_dir).rglob("*.mp4"))
    optic_flow_method = optic_flow_method.lower()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if optic_flow_method.lower() == "farneback":
        alternative_of = OpticFlowCalculatorFarneback(video_path=alternative_video_path)
        neon_of = OpticFlowCalculatorFarneback(video_path=neon_video_path)

    elif optic_flow_method.lower() == "lucas-kanade":
        alternative_of = OpticFlowCalculatorLK(video_path=alternative_video_path)
        neon_of = OpticFlowCalculatorLK(video_path=neon_video_path)
    else:
        raise ValueError(
            'Invalid optic flow choice. Choose from "farneback" or "lucas-kanade"'
        )

    optic_flow_neon = neon_of.process_video(
        output_file_path=Path(output_dir, f"neon_optic_flow.csv")
    )
    optic_flow_alternative = alternative_of.process_video(
        output_file_path=Path(output_dir, f"alternative_optic_flow.csv")
    )

    return optic_flow_neon, optic_flow_alternative

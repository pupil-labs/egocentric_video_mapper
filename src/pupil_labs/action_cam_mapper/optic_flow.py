import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass,asdict,field
from abc import ABC, abstractmethod
from utils import VideoHandler


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
    def __init__(self, video_dir):
        self.video_handler = VideoHandler(video_dir)
        self.optic_flow_result = pd.DataFrame.from_dict(
            dict(start=[], end=[], dx=[], dy=[], angle=[]))

    def process_video(self, start_time=None, end_time=None, output_file=None):
        """Method to calculate the optic flow in a defined video interval. Optic flow is calculated between consecutive frames in the interval.
        Args:
            start_time (float): start time of the video interval to calculate the optic flow signal. Defaults to None. If not specified, the start time is the first timestamp in the video.
            end_time (float): end time of the video interval to calculate the optic flow signal. Defaults to None. If not specified, the end time is the last timestamp in the video.
            output_file (str, optional): Path to a csv file to store the calculated optic flow signal. Defaults to None. If not specified, the signal is not saved to a file. 

        Returns:
            DataFrame: Pandas DataFrame containing the calculated optic flow signal. Has the following columns: 'start', 'end', 'avg_displacement_x', 'avg_displacement_y', 'angle'. 
        """
        if start_time is None:
            start_time = self.video_handler.timestamps[0]
        if end_time is None:
            end_time = self.video_handler.timestamps[-1]
        selected_timestamps = self.video_handler.get_timestamps_in_interval(
            start_time, end_time)
        
        requested_optic_flow = dict(start=[], end=[], dx=[
        ], dy=[], angle=[])
        for ts1, ts2 in zip(selected_timestamps[:-1], selected_timestamps[1:]):
            if self._is_optic_flow_already_calculated(ts1, ts2):
                flow = self._retrieve_optic_flow(ts1, ts2)
            else:
                frame1, frame2 = self.video_handler.get_frame_by_timestamp(ts1), self.video_handler.get_frame_by_timestamp(ts2)
                flow = self._calculate_optical_flow_between_frames(frame1, frame2, ts1, ts2)
            for key,value in asdict(flow).items():
                requested_optic_flow[key].append(value)
        requested_optic_flow = pd.DataFrame.from_dict(
            requested_optic_flow, orient='columns')
        self.optic_flow_result = pd.concat(
            [self.optic_flow_result if not self.optic_flow_result.empty else None, requested_optic_flow], ignore_index=True)
        self.optic_flow_result.drop_duplicates(subset=['start','end'], keep='last', inplace=True)
        self.optic_flow_result.sort_values(by=['start'], inplace=True, ignore_index=True)
        if output_file:
            self.write_to_csv(output_file, requested_optic_flow)
        return requested_optic_flow

    def _is_optic_flow_already_calculated(self, start_timestamp, end_timestamp):
        return not self.optic_flow_result[(self.optic_flow_result['start'] == start_timestamp) & (self.optic_flow_result['end'] == end_timestamp)].empty

    def _retrieve_optic_flow(self, ts1, ts2):
        optic_flow_query = self.optic_flow_result[(
            self.optic_flow_result['start'] == ts1) & (self.optic_flow_result['end'] == ts2)].drop(columns=['angle'])
        return OpticFlowResult(**optic_flow_query.to_dict(orient='records')[0]) if not optic_flow_query.empty else None

    @abstractmethod
    def _calculate_optical_flow_between_frames(self, first_frame, second_frame, first_ts=None,second_ts=None) -> dict:
        return

    def write_to_csv(self,output_file, optic_flow_data=None):
        """Method to write the optic flow data to a csv file. If the file already exists, the data is appended to it, duplicates are removed and the file is sorted by the 'start' column to ensure the data is in chronological order.
        Args:
            output_file (str): Path to a csv file to store the calculated optic flow signal.
            optic_flow_data (DataFrame, optional): Pandas DataFrame containing an optic flow signal. Defaults to None. If not specified, the optic flow signal stored in optic_flow_result attribute is used.
        """
        if optic_flow_data is None:
            optic_flow_data = self.optic_flow_result
        if Path(output_file).exists():
            print('File already exists, appending to it') 
            optic_flow_file = pd.read_csv(output_file, dtype=np.float32)
            optic_flow_data = pd.concat([optic_flow_file, optic_flow_data],ignore_index=True)
        optic_flow_data.drop_duplicates(subset=['start','end'], keep='last', inplace=True)
        optic_flow_data.sort_values(by=['start'], inplace=True, ignore_index=True)
        optic_flow_data.to_csv(output_file, index=False)


class OpticFlowCalculatorLK(OpticFlowCalculatorBase):
    """ Class to calculate optic flow using the Lucas-Kanade method. Backward and forward flow are calculated and compared to ensure the accuracy of points found in the forward flow.

    Args:
        video_dir (str): Path to the video file.
        grid_spacing (int, optional): Spacing between grid points to track. Defaults to 50. The smaller the spacing, the more points are tracked and the more time expensive the calculation is.
    """
    def __init__(self, video_dir, grid_spacing=50, params={'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)}):
        super().__init__(video_dir)
        self.grid_spacing = grid_spacing
        self.lk_params = params
        self.grid_points = self._create_grid_points()

    def _create_grid_points(self):
        xx, yy = np.mgrid[0:self.video_handler.width:self.grid_spacing, 0:self.video_handler.height:self.grid_spacing]
        point_coordinates = np.array(
            [[xx[i, j], yy[i, j]] for i in range(xx.shape[0]) for j in range(xx.shape[1])])
        return point_coordinates.reshape(-1, 1, 2).astype(np.float32)

    def _calculate_optical_flow_between_frames(self, first_frame, second_frame,first_ts=None,second_ts=None):
        first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        second_frame = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)

        p1, status, err = cv.calcOpticalFlowPyrLK(
            first_frame, second_frame, self.grid_points, None, **self.lk_params)
        p0r, status, err = cv.calcOpticalFlowPyrLK(
            second_frame, first_frame, p1, None, **self.lk_params)
        
        chebyshev_distance = abs(self.grid_points-p0r).reshape(-1, 2).max(-1)
        status = chebyshev_distance < 1.0
        good_new_points, good_old_points = p1[status], self.grid_points[status]

        displacement = good_new_points.reshape(-1,2) - good_old_points.reshape(-1,2)

        return OpticFlowResult(dx=np.mean(displacement[:, 0]), dy=np.mean(displacement[:, 1]),start=first_ts,end=second_ts)


class OpticFlowCalculatorFarneback(OpticFlowCalculatorBase):
    """ Class to calculate dense optic flow using the  Gunnar Farneback's algorithm.

    Args:
        video_dir (str): Path to the video file.
    """
    def __init__(self, video_dir, params={'pyr_scale': 0.5, 'levels': 3, 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0}):
        super().__init__(video_dir)
        self.farneback_params = params

    def _calculate_optical_flow_between_frames(self, first_frame, second_frame,first_ts=None,second_ts=None):
        dense_optical_flow = cv.calcOpticalFlowFarneback(cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY), cv.cvtColor(
            second_frame, cv.COLOR_BGR2GRAY), None, **self.farneback_params)
        
        avg_displacement_x = np.mean(dense_optical_flow[:, :, 0])
        avg_displacement_y = np.mean(dense_optical_flow[:, :, 1])

        return OpticFlowResult(dx=avg_displacement_x, dy=avg_displacement_y,start=first_ts,end=second_ts)

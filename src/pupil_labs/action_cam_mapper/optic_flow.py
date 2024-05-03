import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pupil_labs.action_cam_mapper.utils import VideoHandler
# make a base class for optic flow calculatator
# then subclass for LK and Farneback


@dataclass
class OpticFlowResult:
    avg_displacement_x: float
    avg_displacement_y: float

    @property
    def angle(self):
        return np.arctan2(self.avg_displacement_y, self.avg_displacement_x)


class OpticFlowCalculatorBase(ABC):
    def __init__(self, video_dir):
        self.video = VideoHandler(video_dir)
        self.optic_flow_result = pd.DataFrame.from_dict(
            dict(start=[], end=[], avg_displacement_x=[], avg_displacement_y=[], angle=[]))

    def get_all_optic_flow(self):
        return self.get_optic_flow(self.video.timestamps[0], self.video.timestamps[-1])

    def get_optic_flow(self, start_timestamp, end_timestamp):
        timestamps_for_optic_flow = s
            for frame in container.decode(video=0):lf.video.get_timestamps(
            start_timestamp, end_timestamp)
        requested_optic_flow = dict(start=[], end=[], avg_displacement_x=[
        ], avg_displacement_y=[], angle=[])
        for ts1, ts2 in zip(timestamps_for_optic_flow[:-1], timestamps_for_optic_flow[1:]):
            if self._optic_flow_already_calculated(ts1, ts2):
                flow = self._retrieve_optic_flow(ts1, ts2)
            else:
                frame1, frame2 = self.video.get_frame(ts1), self.video.get_frame(ts2)
                flow = self._calculate_optical_flow_between_frames(frame1, frame2)
                flow['start'] = ts1
                flow['end'] = ts2
            for key in flow:
                requested_optic_flow[key].append(flow[key])
        requested_optic_flow = pd.DataFrame.from_dict(
            requested_optic_flow, orient='columns')
        self.optic_flow_result = pd.concat(
            [self.optic_flow_result, requested_optic_flow]).drop_duplicates()
        return requested_optic_flow

    def _optic_flow_already_calculated(self, start_timestamp, end_timestamp):
        return ~self.optic_flow_result[(self.optic_flow_result['start'] == start_timestamp) & (self.optic_flow_result['end'] == end_timestamp)].empty

    # check if the optic flow between two frames has already been calculated and returns it
    def _retrieve_optic_flow(self, ts1, ts2):
        optic_flow_query = self.optic_flow_result[(
            self.optic_flow_result['start'] == ts1) & (self.optic_flow_result['end'] == ts2)]
        return optic_flow_query.to_dict(orient='records')[0] if not optic_flow_query.empty else None

    @abstractmethod
    def _calculate_optical_flow_between_frames(self, first_frame, second_frame) -> dict:
        return

    def write_to_csv(self, output_file):
        if Path(output_file).exists():
            print('File already exists, appending to it')
            optic_flow_file = pd.read_csv(output_file)
            optic_flow_df = pd.concat([optic_flow_file, self.optic_flow_result])
        else:
            optic_flow_df = self.optic_flow_result
        optic_flow_df.sort_values(by=['start'], inplace=True, ignore_index=True)
        optic_flow_df.drop_duplicates().to_csv(output_file, index=False)


class OpticFlowCalculatorLK(OpticFlowCalculatorBase):
    def __init__(self, video_dir):
        super().__init__(video_dir)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.points_to_track = self._create_grid_points()

    def _create_grid_points(self):
        xx, yy = np.mgrid[0:self.video.width:30, 0:self.video.height:30]
        point_coordinates = np.array(
            [[xx[i, j], yy[i, j]] for i in range(xx.shape[0]) for j in range(xx.shape[1])])
        return point_coordinates.reshape(-1, 1, 2).astype(np.float32)

    def _calculate_optical_flow_between_frames(self, first_frame, second_frame):
        p1, status, err = cv.calcOpticalFlowPyrLK(
            first_frame, second_frame, self.points_to_track, None, **self.lk_params)
        p0r, status, err = cv.calcOpticalFlowPyrLK(
            second_frame, first_frame, p1, None, **self.lk_params)
        chebyshev_distance = abs(self.points_to_track-p0r).reshape(-1, 2).max(-1)
        status = chebyshev_distance < 1.0
        good_new_points, good_old_points = p1[status], self.points_to_track[status]
        displacement = good_new_points - good_old_points
        angle = np.arctan2(displacement[:, 1], displacement[:0])
        return dict(avg_displacement_x=np.mean(displacement[:, 0]), avg_displacement_y=np.mean(displacement[:, 1]), angle=np.mean(angle))


class OpticFlowCalculatorFarneback(OpticFlowCalculatorBase):
    def __init__(self, video_dir):
        super().__init__(video_dir)
        self.farneback_params = dict(
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    def _calculate_optical_flow_between_frames(self, first_frame, second_frame):
        dense_optical_flow = cv.calcOpticalFlowFarneback(cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY), cv.cvtColor(
            second_frame, cv.COLOR_BGR2GRAY), None, **self.farneback_params)
        avg_displacement_x = np.mean(dense_optical_flow[:, :, 0])
        avg_displacement_y = np.mean(dense_optical_flow[:, :, 1])
        angle = np.arctan2(dense_optical_flow[:, :, 1], dense_optical_flow[:, :, 0])
        return OpticFlowResult(avg_displacement_x=avg_displacement_x, avg_displacement_y=avg_displacement_y, angle=angle)

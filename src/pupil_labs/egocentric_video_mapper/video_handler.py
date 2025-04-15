import logging

import numpy as np
import pupil_labs.video as plv

logger = logging.getLogger(__name__)


class VideoHandler:
    def __init__(self, video_path):
        self.path = video_path
        self.video_container = plv.Reader(video_path)
        self._timestamps = self._get_timestamps()
        self._set_properties()

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def fps(self):
        return self._fps

    def get_frame_by_timestamp(self, timestamp):
        timestamp, ts_idx = self.get_closest_timestamp(timestamp)
        frame = self.video_container[ts_idx].bgr[:, :, :3]
        return frame

    def _get_timestamps(self):
        self.pts = self.video_container.pts
        return np.array(self.video_container.container_timestamps, dtype=np.float32)

    def _set_properties(self):
        self._height = self.video_container.height
        self._width = self.video_container.width
        self._fps = self.video_container.average_rate
        self._read_frames = 0
        self.lpts = -1

    def get_timestamps_in_interval(self, start_time=0, end_time=np.inf):
        assert (
            start_time < end_time
        ), f"Start time ({start_time} s) must be smaller than end time ({end_time} s)"
        return self.timestamps[
            (self.timestamps >= start_time) & (self.timestamps <= end_time)
        ]

    def get_closest_timestamp(self, time):
        after_index = np.searchsorted(self.timestamps, time)
        if after_index == len(self.timestamps):
            after_index = len(self.timestamps) - 1
        before_index = after_index - 1

        ts_after = self.timestamps[after_index]
        ts_before = self.timestamps[before_index]

        if np.abs(ts_after - time) < np.abs(ts_before - time):
            return ts_after, int(after_index)
        else:
            return ts_before, int(before_index)

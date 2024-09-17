import logging

import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoHandler:
    def __init__(self, video_path):
        self.path = video_path
        self.video_container = av.open(self.path)
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
        pts = self.pts[ts_idx]
        # if seeking backwards, reset the video container
        if pts < self.lpts:
            logger.info("Seeking backwards, resetting video container")
            self._read_frames = 0
            self.video_container = av.open(self.path)

        if self._read_frames == 0:
            self.lpts = -1
            self.last_frame = None

        vid_frame, self.lpts = get_frame(
            self.video_container, pts, self.lpts, self.last_frame
        )
        frame = vid_frame.to_ndarray(format="rgb24")
        self.last_frame = vid_frame
        self._read_frames += 1
        return frame

    def _get_timestamps(self):
        container = av.open(self.path)
        video = container.streams.video[0]
        self.pts = [
            packet.pts for packet in container.demux(video) if packet.pts is not None
        ]
        av_timestamps = (
            np.asarray(self.pts, dtype=np.int32) / video.time_base.denominator
        )
        container.close()
        av_timestamps.sort()
        return np.asarray(av_timestamps, dtype=np.float32)

    def _set_properties(self):
        container = av.open(self.path)
        video = container.streams.video[0]
        self._height = video.height
        self._width = video.width
        self._fps = float(video.average_rate)
        self._time_base = video.time_base
        container.close()
        self._read_frames = 0
        self.lpts = -1

    def get_timestamps_in_interval(self, start_time=0, end_time=np.inf):
        assert (
            start_time < end_time
        ), f"Start time ({start_time} s) must be smaller than end time ({end_time} s)"
        return self.timestamps[
            (self.timestamps >= start_time) & (self.timestamps <= end_time)
        ]

    def get_closest_timestamp(self, time):  # debug
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


def get_frame(av_container, pts, last_pts, frame, audio=False):
    """Gets the frame at the given timestamp.
    :param av_container: The container of the video.
    :param pts: The pts of the frame we are looking for.
    :param last_pts: The last pts of the video readed.
    :param frame: Last frame decoded.

    Args:
        av_container : The container of the video.
        pts (int):  The pts of the frame we are looking for.
        last_pts (int): The last pts of the video readed.
        frame (av.video.frame.VideoFrame): Last frame decoded.
        audio (bool, optional): Whether if it is audio or not. Defaults to False.

    """
    if audio:
        strm = av_container.streams.audio[0]
    else:
        strm = av_container.streams.video[0]
    if last_pts < pts:
        try:
            for frame in av_container.decode(strm):
                logging.debug(
                    f"Frame {frame.pts} read from video and looking for {pts}"
                )
                if pts == frame.pts:
                    last_pts = frame.pts
                    return frame, last_pts
                if pts < frame.pts:
                    logging.warning(f"Frame {pts} not found in video, used {frame.pts}")
                    last_pts = frame.pts
                    return frame, last_pts
        except av.EOFError:
            logging.info("End of the file")
            return None, last_pts
    else:
        logging.debug("This frame was already decoded")
        return frame, last_pts

import logging
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
from pupil_labs.dynamic_content_on_rim.video.read import get_frame


class VideoHandler:
    def __init__(self, video_path):
        self.path = video_path
        self.video_container = av.open(self.path)
        self._timestamps = self.get_timestamps()
        self.set_properties()

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
        timestamp, _ = self.get_closest_timestamp(timestamp)
        pts = np.int32(np.round(timestamp * self._time_base.denominator))

        if pts < self.lpts:  # if seeking backwards, reset the video container
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

    def get_timestamps(self):
        container = av.open(self.path)
        video = container.streams.video[0]
        av_timestamps = [
            packet.pts / video.time_base.denominator
            for packet in container.demux(video)
            if packet.pts is not None
        ]
        container.close()
        av_timestamps.sort()
        return np.asarray(av_timestamps, dtype=np.float32)

    def set_properties(self):
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

    def get_surrounding_timestamps(self, time):
        closest_timestamp = self.get_closest_timestamp(time)

        if closest_timestamp > time:
            previous_timestamp = self.timestamps[
                np.where(self.timestamps == closest_timestamp)[0][0] - 1
            ]
            next_timestamp = closest_timestamp
        else:
            previous_timestamp = closest_timestamp
            next_timestamp = self.timestamps[
                np.where(self.timestamps == closest_timestamp)[0][0] + 1
            ]

        return previous_timestamp, next_timestamp


def write_timestamp_csv(neon_timeseries_dir, aligned_relative_ts, output_file_dir=None):
    """Function that creates a timestamp csv file for the alternative camera recording in the same format as the world_timestamps.csv. The csv file is saved in the same directory as the world_timestamps.csv of the given Neon recording.

    Args:
        neon_timeseries_dir (str): Path to the directory of the Neon recording containing the world_timestamps.csv
        aligned_relative_ts (ndarray): Timestamps of the alternative camera recording, obtained from the metadata of the video file. This function assumes that the timestamps are already aligned with the relative Neon recording timestamps (offset is corrected).
        output_file_dir (str, optional): Path to the directory where thealternative camera timestamps csv file will be saved. If None, the file is saved in the same directory as the world_timestamps.csv. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    output_file_path = Path(
        neon_timeseries_dir if output_file_dir is None else output_file_dir,
        "alternative_camera_timestamps.csv",
    )

    neon_timestamps_path = Path(neon_timeseries_dir, "world_timestamps.csv")
    if not neon_timestamps_path.exists():
        raise FileNotFoundError(
            f"world_timestamps.csv not found in {neon_timeseries_dir}, please make sure the file exists"
        )

    neon_timestamps_df = pd.read_csv(neon_timestamps_path)
    columns_for_mapping = neon_timestamps_df.columns

    alternative_cam_timestamps = np.int64(aligned_relative_ts / 1e-9)
    alternative_cam_timestamps += neon_timestamps_df["timestamp [ns]"].iloc[0]

    alternative_timestamps_df = pd.DataFrame.from_dict(
        {col: [None for _ in alternative_cam_timestamps] for col in columns_for_mapping}
    )
    alternative_timestamps_df["timestamp [ns]"] = alternative_cam_timestamps
    alternative_timestamps_df["recording id"] = neon_timestamps_df[
        "recording id"
    ].values[0]

    first_ts = neon_timestamps_df["timestamp [ns]"].values[0]
    last_ts = neon_timestamps_df["timestamp [ns]"].values[-1]

    mssg = f'First timestamp in alternative camera recording ({pd.Timestamp(alternative_timestamps_df["timestamp [ns]"].values[0],unit="ns")}) is {"before" if alternative_timestamps_df["timestamp [ns]"].values[0]<first_ts else "after"} the first timestamp of Neon Scene recording ({pd.Timestamp(first_ts,unit="ns")})'
    (
        logger.warning(mssg)
        if alternative_timestamps_df["timestamp [ns]"].values[0] < first_ts
        else logger.info(mssg)
    )

    mssg = f'Last timestamp of alternative camera recording ({pd.Timestamp(alternative_timestamps_df["timestamp [ns]"].values[-1],unit="ns")}) is {"before" if alternative_timestamps_df["timestamp [ns]"].values[-1]<last_ts else "after"} the last timestamp of Neon Scene recording ({pd.Timestamp(last_ts, unit="ns")})'
    (
        logger.warning(mssg)
        if alternative_timestamps_df["timestamp [ns]"].values[-1] > last_ts
        else logger.info(mssg)
    )

    for section in neon_timestamps_df["section id"].unique():
        start_section = min(
            neon_timestamps_df[neon_timestamps_df["section id"] == section][
                "timestamp [ns]"
            ]
        )
        end_section = max(
            neon_timestamps_df[neon_timestamps_df["section id"] == section][
                "timestamp [ns]"
            ]
        )
        alternative_timestamps_df.loc[
            alternative_timestamps_df["timestamp [ns]"].between(
                start_section, end_section, inclusive="left"
            ),
            "section id",
        ] = section

    alternative_timestamps_df.loc[
        (alternative_timestamps_df["section id"].isnull())
        & (alternative_timestamps_df["timestamp [ns]"] < first_ts),
        "section id",
    ] = neon_timestamps_df.loc[
        neon_timestamps_df["timestamp [ns]"] == first_ts, "section id"
    ].values[
        0
    ]
    alternative_timestamps_df.loc[
        (alternative_timestamps_df["section id"].isnull())
        & (alternative_timestamps_df["timestamp [ns]"] >= last_ts),
        "section id",
    ] = neon_timestamps_df.loc[
        neon_timestamps_df["timestamp [ns]"] == last_ts, "section id"
    ].values[
        0
    ]

    alternative_timestamps_df.to_csv(output_file_path, index=False)
    mssg = f"Timestamps for alternative camera recording saved at {output_file_path}"
    logger.info(mssg)
    print(mssg)


def generate_mapper_kwargs(
    neon_timeseries_dir,
    alternative_vid_path,
    output_dir,
    matcher_choice,
    optical_flow_method,
):
    matcher_choice = matcher_choice.lower()
    image_matcher_parameters = {
        "efficient_loftr": {"model_type": "opt", "gpu_num": 0},
        "loftr": {"location": "indoor", "gpu_num": 0},
        "disk_lightglue": {"num_features": 2048, "gpu_num": 0},
        "dedode_lightglue": {"num_features": 5000, "gpu_num": 0},
    }
    neon_vid_path = Path(neon_timeseries_dir).rglob("*.mp4").__next__()

    alternative_timestamps_path = Path(output_dir, "alternative_camera_timestamps.csv")
    if not alternative_timestamps_path.exists():
        alternative_timestamps_path = Path(
            neon_timeseries_dir, "alternative_camera_timestamps.csv"
        )
    if not alternative_timestamps_path.exists():
        raise FileNotFoundError(
            f"Alternative camera timestamps file not found, please make sure the file exists either in the output directory ({output_dir}) or in the Neon timeseries directory ({neon_timeseries_dir})"
        )

    optic_flow_output_dir = Path(output_dir, "optic_flow")
    method = "lk" if optical_flow_method.lower() == "lucas-kanade" else "farneback"
    mapper_kwargs = {
        "neon_gaze_csv": Path(neon_timeseries_dir, "gaze.csv"),
        "neon_video_path": neon_vid_path,
        "neon_timestamps": Path(neon_timeseries_dir, "world_timestamps.csv"),
        "neon_opticflow_csv": Path(optic_flow_output_dir, f"neon_{method}_of.csv"),
        "alternative_video_path": alternative_vid_path,
        "alternative_timestamps": alternative_timestamps_path,
        "alternative_opticflow_csv": Path(
            optic_flow_output_dir, f"alternative_{method}_of.csv"
        ),
        "image_matcher": matcher_choice,
        "image_matcher_parameters": image_matcher_parameters[matcher_choice],
        "output_dir": Path(output_dir, f"mapped_gaze/{matcher_choice}_{method}"),
        "patch_size": 1000,
        "verbose": False,
    }
    return mapper_kwargs


def generate_comparison_video_kwargs(
    neon_timeseries_dir,
    alternative_vid_path,
    mapped_gaze_path,
    output_dir,
):
    alternative_gaze_dict = {"Alternative Egocentric View": Path(mapped_gaze_path)}
    neon_vid_path = Path(neon_timeseries_dir).rglob("*.mp4").__next__()

    rendered_video_path = Path(
        output_dir,
        f"rendered_videos/neon_comparison_{Path(mapped_gaze_path).parent.stem}.mp4",
    )
    Path(rendered_video_path).parent.mkdir(parents=True, exist_ok=True)

    comparison_video_args = {
        "alternative_video_path": alternative_vid_path,
        "alternative_timestamps_path": Path(
            output_dir, "alternative_camera_timestamps.csv"
        ),
        "alternative_gaze_paths_dict": alternative_gaze_dict,
        "neon_video_path": neon_vid_path,
        "neon_worldtimestamps_path": Path(neon_timeseries_dir, "world_timestamps.csv"),
        "neon_gaze_path": Path(neon_timeseries_dir, "gaze.csv"),
        "save_video_path": rendered_video_path,
        "same_frame": False,
    }
    return comparison_video_args


def get_file(folder_dir, file_suffix=".mp4", required_in_name="0"):
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder_dir)
        for name in files
        if name.endswith(file_suffix) and required_in_name in name
    ][0]

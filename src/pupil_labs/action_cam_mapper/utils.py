import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pupil_labs.video as plv


class VideoHandler:
    def __init__(self, video_path):
        self.path = video_path
        self.video_stream = self.open_video()
        self._timestamps = self.get_timestamps()

    @property
    def height(self):
        return self.video_stream.height

    @property
    def width(self):
        return self.video_stream.width

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def fps(self):
        return (
            self.video_stream.average_rate.numerator
            / self.video_stream.average_rate.denominator
        )

    def open_video(self):
        container = plv.open(self.path)
        video_stream = container.streams.video[0]
        if video_stream.type != "video":
            raise ValueError(f"No video stream found in {self.path}")
        video_stream.logger.setLevel(logging.ERROR)
        return video_stream

    def close_video(self):
        self.video_stream.close()

    def get_timestamps(self):
        video_timestamps = np.asarray(self.video_stream.pts)
        video_timestamps = video_timestamps / self.video_stream.time_base.denominator
        return np.asarray(video_timestamps, dtype=np.float32)

    def get_frame_by_timestamp(self, timestamp):
        timestamp, timestamp_index = self.get_closest_timestamp(timestamp)
        frame = self.video_stream.frames[timestamp_index]
        frame = frame.to_image()
        return np.asarray(frame)

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


def write_timestamp_csv(neon_timeseries_path, aligned_relative_ts, saving_path=None):
    """Function that creates a timestamp csv file for the alternative camera recording in the same format as the world_timestamps.csv. The csv file is saved in the same directory as the world_timestamps.csv of the given Neon recording.

    Args:
        neon_timeseries_path (str): Path to the directory of the Neon recording containing the world_timestamps.csv
        aligned_relative_ts (ndarray): Timestamps of the alternative camera recording, obtained from the metadata of the video file. This function assumes that the timestamps are already aligned with the relative Neon recording timestamps (offset is corrected).
        saving_path (str, optional): Path to save the alternative camera timestamps csv file. If None, the file is saved in the same directory as the world_timestamps.csv. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    neon_timestamps_path = Path(neon_timeseries_path, "world_timestamps.csv")
    if not neon_timestamps_path.exists():
        raise FileNotFoundError(
            f"world_timestamps.csv not found in {neon_timeseries_path}, please make sure the file exists"
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

    saving_path = (
        Path(neon_timestamps_path).parent if saving_path is None else Path(saving_path)
    )
    alternative_timestamps_df.to_csv(
        Path(saving_path, "alternative_camera_timestamps.csv"), index=False
    )
    mssg = f"Timestamps for alternative camera recording saved at {Path(saving_path/'alternative_camera_timestamps.csv')}"
    logger.info(mssg)
    print(mssg)


def generate_mapper_kwargs(
    neon_timeseries_path, alternative_vid_path, output_dir, matcher_choice
):

    image_matcher_parameters = {
        "Efficient_LOFTR": {"model_type": "opt", "gpu_num": 0},
        "LOFTR": {"location": "indoor", "gpu_num": 0},
        "DISK_LightGlue": {"num_features": 2048, "gpu_num": 0},
        "DeDoDe_LightGlue": {"num_features": 5000, "gpu_num": 0},
    }

    optic_flow_output_dir = Path(output_dir, "optic_flow")
    neon_vid_path = Path(neon_timeseries_path).rglob("*.mp4").__next__()
    mapper_kwargs = {
        "neon_gaze_csv": Path(neon_timeseries_path, "gaze.csv"),
        "neon_video_dir": neon_vid_path,
        "alternative_video_dir": alternative_vid_path,
        "neon_timestamps": Path(neon_timeseries_path, "world_timestamps.csv"),
        "alternative_timestamps": Path(output_dir, "alternative_camera_timestamps.csv"),
        "neon_opticflow_csv": Path(optic_flow_output_dir, "neon_lk_of.csv"),
        "alternative_opticflow_csv": Path(
            optic_flow_output_dir, "alternative_lk_of.csv"
        ),
        "image_matcher": matcher_choice,
        "image_matcher_parameters": image_matcher_parameters[matcher_choice],
        "patch_size": 1000,
        "verbose": False,
    }
    return mapper_kwargs


def generate_comparison_video_kwargs(
    neon_timeseries_path,
    alternative_vid_path,
    mapped_gaze_path,
    output_dir,
    image_matcher_choice,
):
    alternative_gaze_dict = {"Alternative Egocentric View": Path(mapped_gaze_path)}
    neon_vid_path = Path(neon_timeseries_path).rglob("*.mp4").__next__()

    rendered_video_path = Path(
        output_dir,
        f"rendered_videos/neon_comparison_{image_matcher_choice.lower()}_lk.mp4",
    )
    Path(rendered_video_path).parent.mkdir(parents=True, exist_ok=True)

    comparison_video_args = {
        "alternative_video_path": alternative_vid_path,
        "alternative_timestamps_path": Path(
            output_dir, "alternative_camera_timestamps.csv"
        ),
        "alternative_gaze_paths_dict": alternative_gaze_dict,
        "neon_video_path": neon_vid_path,
        "neon_worldtimestamps_path": Path(neon_timeseries_path, "world_timestamps.csv"),
        "neon_gaze_path": Path(neon_timeseries_path, "gaze.csv"),
        "save_video_path": rendered_video_path,
        "same_frame": False,
    }
    return comparison_video_args


def get_file(folder_path, file_suffix=".mp4", required_in_name="0"):
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder_path)
        for name in files
        if name.endswith(file_suffix) and required_in_name in name
    ][0]

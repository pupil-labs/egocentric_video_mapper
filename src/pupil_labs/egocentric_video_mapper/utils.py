import logging
import os
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pupil_labs.egocentric_video_mapper.video_handler import VideoHandler

logger = logging.getLogger(__name__)


def write_timestamp_csv(neon_timeseries_dir, aligned_relative_ts, output_file_dir=None):
    """Function that creates a timestamp csv file for the alternative camera recording in the same format as the world_timestamps.csv. The csv file is saved in the same directory as the world_timestamps.csv of the given Neon recording.

    Args:
        neon_timeseries_dir (str): Path to the directory of the Neon recording containing the world_timestamps.csv
        aligned_relative_ts (ndarray): Timestamps of the alternative camera recording, obtained from the metadata of the video file. This function assumes that the timestamps are already aligned with the relative Neon recording timestamps (offset is corrected).
        output_file_dir (str, optional): Path to the directory where thealternative camera timestamps csv file will be saved. If None, the file is saved in the same directory as the world_timestamps.csv. Defaults to None.
    """
    output_file_path = (
        Path(output_file_dir or neon_timeseries_dir)
        / "alternative_camera_timestamps.csv"
    )

    neon_timestamps_path = Path(neon_timeseries_dir, "world_timestamps.csv")
    if not neon_timestamps_path.exists():
        raise FileNotFoundError(
            f"world_timestamps.csv not found in {neon_timeseries_dir}, please make sure the file exists"
        )

    neon_timestamps_df = pd.read_csv(neon_timestamps_path)
    columns_for_mapping = neon_timestamps_df.columns

    alternative_cam_timestamps = np.int64(aligned_relative_ts * 1e9)
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

    if alternative_timestamps_df["timestamp [ns]"].values[0] < first_ts:
        time_relation = "before"
    else:
        time_relation = "after"

    mssg = f'First timestamp in alternative camera recording ({pd.Timestamp(alternative_timestamps_df["timestamp [ns]"].values[0],unit="ns")}) is {time_relation} the first timestamp of Neon Scene recording ({pd.Timestamp(first_ts,unit="ns")})'

    if alternative_timestamps_df["timestamp [ns]"].values[0] < first_ts:
        logger.warning(mssg)
    else:
        logger.info(mssg)

    if alternative_timestamps_df["timestamp [ns]"].values[-1] < last_ts:
        time_relation = "before"
    else:
        time_relation = "after"

    mssg = f'Last timestamp of alternative camera recording ({pd.Timestamp(alternative_timestamps_df["timestamp [ns]"].values[-1],unit="ns")}) is {time_relation} the last timestamp of Neon Scene recording ({pd.Timestamp(last_ts, unit="ns")})'

    if alternative_timestamps_df["timestamp [ns]"].values[-1] > last_ts:
        logger.warning(mssg)
    else:
        logger.info(mssg)

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


def generate_mapper_kwargs(
    neon_timeseries_dir,
    alternative_vid_path,
    output_dir,
    matcher_choice,
    logging_level="WARNING",
):
    matcher_choice = matcher_choice.lower()
    image_matcher_parameters = {
        "efficient_loftr": {"model_type": "opt", "gpu_num": 0},
        "loftr_indoor": {"location": "indoor", "gpu_num": 0},
        "loftr_outdoor": {"location": "outdoor", "gpu_num": 0},
        "disk_lightglue": {"num_features": 2048, "gpu_num": 0},
        "dedode_lightglue": {"num_features": 5000, "gpu_num": 0},
    }

    alternative_timestamps_path = Path(output_dir, "alternative_camera_timestamps.csv")
    if not alternative_timestamps_path.exists():
        alternative_timestamps_path = Path(
            neon_timeseries_dir, "alternative_camera_timestamps.csv"
        )
    if not alternative_timestamps_path.exists():
        raise FileNotFoundError(
            f"Alternative camera timestamps file not found."
            f"Please make sure the file exists either in the output directory ({output_dir}) or in the Neon timeseries directory ({neon_timeseries_dir})"
        )

    mapper_kwargs = {
        "neon_timeseries_dir": Path(neon_timeseries_dir),
        "alternative_video_path": alternative_vid_path,
        "alternative_timestamps": alternative_timestamps_path,
        "image_matcher": matcher_choice,
        "image_matcher_parameters": image_matcher_parameters[matcher_choice],
        "neon_opticflow_csv": Path(output_dir, f"neon_optic_flow.csv"),
        "alternative_opticflow_csv": Path(output_dir, f"alternative_optic_flow.csv"),
        "output_dir": Path(output_dir),
        "patch_size": 1000,
        "logging_level": logging_level,
    }
    return mapper_kwargs


def generate_comparison_video_kwargs(
    neon_timeseries_dir,
    alternative_vid_path,
    mapped_gaze_path,
    output_dir,
):
    alternative_gaze_dict = {"Alternative Egocentric View": Path(mapped_gaze_path)}
    neon_vid_path = next(Path(neon_timeseries_dir).rglob("*.mp4"))
    rendered_video_path = Path(
        output_dir,
        "alternative_camera-neon_comparison.mp4",
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


def execute_ffmpeg_command(rotation, video_path):
    rotation_map = {
        "90° clockwise": ["-vf", "transpose=1"],
        "90° counterclockwise": ["-vf", "transpose=2"],
        "180°": ["-vf", "transpose=1,transpose=1"],
    }
    name_new_video = (
        os.path.splitext(os.path.basename(video_path))[0] + "_corrected_orientation.mp4"
    )
    name_new_video = os.path.join(os.path.dirname(video_path), name_new_video)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-metadata:s:v:0",
        "rotate=0",
        *rotation_map.get(rotation, []),
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "copy",
        name_new_video,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (
        (" ").join(command),
        name_new_video,
    )


def show_videos_preview(neon_timeseries_dir, alternative_video_path):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    neon_video_path = get_file(neon_timeseries_dir)
    neon_scene_video = VideoHandler(neon_video_path)
    frames_in_video = len(neon_scene_video.timestamps)
    frame_for_display = cv2.cvtColor(
        neon_scene_video.get_frame_by_timestamp(
            neon_scene_video.timestamps[frames_in_video // 2]
        ),
        cv2.COLOR_BGR2RGB,
    )
    plt.imshow(frame_for_display)
    plt.axis("off")
    plt.title("Neon Scene Video")
    # Display the alternative egocentric video
    alternative_egocentric_video = VideoHandler(alternative_video_path)
    frames_in_video = len(alternative_egocentric_video.timestamps)

    frame_for_display = cv2.cvtColor(
        alternative_egocentric_video.get_frame_by_timestamp(
            alternative_egocentric_video.timestamps[frames_in_video // 2]
        ),
        cv2.COLOR_BGR2RGB,
    )
    plt.subplot(1, 2, 2)
    plt.imshow(frame_for_display)
    plt.axis("off")
    plt.title("Alternative Egocentric Video")
    plt.show()

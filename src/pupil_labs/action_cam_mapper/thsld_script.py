import os
import sys
import pandas as pd
import numpy as np
import logging
from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback
from utils import VideoHandler, write_action_timestamp_csv
from sync_videos import OffsetCalculator
from gaze_mapper import (
    ActionCameraGazeMapper,
    ActionCameraGazeMapper2,
    RulesBasedGazeMapper,
)
from video_renderer import save_video
from pathlib import Path


def get_file(folder_path, file_suffix=".mp4", required_in_name="0"):
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder_path)
        for name in files
        if name.endswith(file_suffix) and required_in_name in name
    ][0]


def main_mapper(
    action_vid_path,
    neon_vid_path,
    neon_timestamps,
    action_timestamps,
    neon_gaze_csv,
    neon_opticflow_csv,
    action_opticflow_csv,
    output_dir,
    matcher,
    optic_flow_choice,
    map_options,
):
    output_dir = Path(output_dir, f'mapped_gaze/{matcher["choice"]}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(funcName)s function in %(name)s: %(message)s"
        )
    )
    logging.basicConfig(
        format="[%(levelname)s]  %(funcName)s function in %(name)s (%(asctime)s):  %(message)s",
        handlers=[
            logging.FileHandler(
                Path(
                    output_dir,
                    f"mapping_rf{map_options['refresh_thrshld']}_of{map_options['opticf_threshold']}_gz{map_options['gaze_change_thrshld']}.log",
                )
            ),
            stream_handler,
        ],
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    param = matcher["parameters"]
    print(f'Using {matcher["choice"]} with parameters: {param}')
    mapper = RulesBasedGazeMapper(
        neon_gaze_csv=neon_gaze_csv,
        neon_video_dir=neon_vid_path,
        action_video_dir=action_vid_path,
        neon_timestamps=neon_timestamps,
        action_timestamps=action_timestamps,
        image_matcher=matcher["choice"],
        image_matcher_parameters=param,
        neon_opticflow_csv=neon_opticflow_csv,
        action_opticflow_csv=action_opticflow_csv,
        patch_size=1000,
    )

    gaze_csv_path = Path(output_dir, f"action_gaze_{optic_flow_choice}.csv")
    mapper.map_gaze(saving_path=gaze_csv_path, **map_options)
    return gaze_csv_path


def main(
    action_vid_path,
    neon_timeseries_dir,
    output_dir,
    image_matcher,
    optic_flow_choice="lk",
    render_video=False,
    map_options={
        "refresh_thrshld": 0.5,
        "opticf_threshold": 0.5,
        "gaze_change_thrshld": 0.5,
    },
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    neon_vid_path = get_file(neon_timeseries_dir, file_suffix=".mp4")
    neon_timestamps = neon_timeseries_dir + "/world_timestamps.csv"
    neon_gaze_csv = neon_timeseries_dir + "/gaze.csv"
    action_timestamps = neon_timeseries_dir + "/action_camera_timestamps.csv"
    neon_of_path = str(Path(output_dir).parent.parent) + "/optic_flow/neon_lk_of.csv"
    action_of_path = (
        str(Path(output_dir).parent.parent) + "/optic_flow/action_lk_of.csv"
    )
    print(neon_of_path)
    # Step 3: Map gaze
    action_gaze_csv = main_mapper(
        action_vid_path=action_vid_path,
        neon_vid_path=neon_vid_path,
        neon_timestamps=neon_timestamps,
        action_timestamps=action_timestamps,
        neon_gaze_csv=neon_gaze_csv,
        neon_opticflow_csv=neon_of_path,
        action_opticflow_csv=action_of_path,
        output_dir=output_dir,
        matcher=image_matcher,
        optic_flow_choice=optic_flow_choice,
        map_options=map_options,
    )

    # # Step 4 (Optional): Render simultaneous videos with gaze in both
    # if render_video:
    #     video_path = Path(
    #         output_dir,
    #         f"video_render/Neon_Action_{image_matcher['choice']}_{optic_flow_choice}.mp4",
    #     )
    #     Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    #     save_video(
    #         action_video_path=action_vid_path,
    #         action_worldtimestamps_path=action_timestamps,
    #         action_gaze_paths_dict={image_matcher["choice"].upper(): action_gaze_csv},
    #         neon_video_path=neon_vid_path,
    #         neon_worldtimestamps_path=neon_timestamps,
    #         neon_gaze_path=neon_gaze_csv,
    #         save_video_path=video_path,
    #     )


if __name__ == "__main__":
    action_vid_path = "/users/sof/gaze_mapping/raw_videos/InstaVid/wearingNeon_2m/AVun_20240216_160246_055.mp4"
    neon_timeseries_path = "/users/sof/gaze_mapping/raw_videos/Neon/Raw_Data/2024-02-16_wearingNeon/2024-02-16_15-58-13-6310bec3/"

    neon_timeseries_path = (
        "/users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62/"
    )
    action_vid_path = "/users/sof/video_examples/second_video/20240523_171941_000.mp4"

    output_dir = (
        "/users/sof/action_map_experiments/"  # parent directory for all outputs
    )
    name = Path(action_vid_path).parent.name
    output_dir = Path(output_dir, name)

    param_lg = {"num_features": 2048, "gpu_num": 0}
    param_loftr = {"location": "indoor", "gpu_num": 0}
    image_matcher_loftr = {"choice": "loftr", "parameters": param_loftr}
    image_matcher_lg = {"choice": "disk_lightglue", "parameters": param_lg}

    main(
        action_vid_path=action_vid_path,
        neon_timeseries_dir=neon_timeseries_path,
        output_dir=output_dir,
        image_matcher=image_matcher_loftr,
        optic_flow_choice="lk",
        mapper_choice=1,
        render_video=False,
    )

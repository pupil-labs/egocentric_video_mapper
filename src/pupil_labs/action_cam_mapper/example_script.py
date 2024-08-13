import os
import sys
import pandas as pd
import numpy as np
import logging
from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback
from utils import VideoHandler, write_action_timestamp_csv
from sync_videos import OffsetCalculator
from gaze_mapper import RulesBasedGazeMapper
from video_renderer import save_comparison_video
from pathlib import Path
import argparse


def get_gaze_per_frame(gaze_file, video_timestamps):
    """This function search for the gaze coordinates with the closest world timestamp to the video world timestamps and returns a list of coordinates for every frame in the video

    Args:
        gaze_file (str): Path to the gaze file associated to a video
        video_timestamps (str): Path to the world timestamps to the frames in a video

    Returns:
        list_coords (list): A list containing the x,y coordinates for every entry in the video_timestamps
    """

    scene_timestamps = pd.read_csv(video_timestamps)
    gaze_timestamps = pd.read_csv(gaze_file)
    scene_ns = scene_timestamps["timestamp [ns]"].to_numpy()
    gaze_ns = gaze_timestamps["timestamp [ns]"].to_numpy()
    list_coords = []
    for scene_time in scene_ns:
        gaze_indexing = np.argmin(np.abs(gaze_ns - scene_time))
        coords = gaze_timestamps.iloc[gaze_indexing][["gaze x [px]", "gaze y [px]"]]
        list_coords.append(coords.to_numpy())
    return list_coords


def get_file(folder_path, file_suffix=".mp4", required_in_name="0"):
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder_path)
        for name in files
        if name.endswith(file_suffix) and required_in_name in name
    ][0]


def calc_optic(neon_video, action_video, output_dir, of_choice="farneback"):
    output_dir = Path(output_dir, "optic_flow")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if of_choice.lower() == "farneback":
        action_of = OpticFlowCalculatorFarneback(video_dir=action_video)
        neon_of = OpticFlowCalculatorFarneback(video_dir=neon_video)

    elif of_choice.lower() == "lk":
        action_of = OpticFlowCalculatorLK(video_dir=action_video)
        neon_of = OpticFlowCalculatorLK(video_dir=neon_video)
    else:
        raise ValueError('Invalid optic flow choice. Choose from "farneback" or "lk"')

    optic_flow_neon = neon_of.process_video()
    optic_flow_action = action_of.process_video()

    action_saving_path = Path(output_dir, f"action_{of_choice}_of.csv")
    neon_saving_path = Path(output_dir, f"neon_{of_choice}_of.csv")
    action_of.write_to_csv(output_file=action_saving_path)
    neon_of.write_to_csv(output_file=neon_saving_path)

    return action_saving_path, neon_saving_path


def align_videos(action_result, neon_result, action_vid_path, neon_timestamps):
    offset_calc = OffsetCalculator(
        src=action_result["dy"].values,
        src_timestamps=action_result["start"].values,
        dst=neon_result["dy"].values,
        dst_timestamps=neon_result["start"].values,
        resampling_frequency=500,
    )
    t_offset, pearson_corr = offset_calc.estimate_time_offset()
    actionVid = VideoHandler(action_vid_path)
    write_action_timestamp_csv(neon_timestamps, actionVid.timestamps + t_offset)


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
    thresholds={
        "refresh_time_thrshld": None,
        "optic_flow_thrshld": None,
        "gaze_change_thrshld": None,
    },
):
    output_dir = Path(output_dir, f'mapped_gaze/{matcher["choice"]}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    mapper.map_gaze(saving_path=gaze_csv_path, **thresholds)
    return gaze_csv_path


def main(
    action_vid_path,
    neon_timeseries_dir,
    output_dir,
    image_matcher,
    optic_flow_choice="lk",
    render_video=False,
    mapper_thresholds={
        "refresh_time_thrshld": None,
        "optic_flow_thrshld": None,
        "gaze_change_thrshld": None,
    },
):

    neon_vid_path = get_file(neon_timeseries_dir, file_suffix=".mp4")
    neon_timestamps = neon_timeseries_dir + "/world_timestamps.csv"
    neon_gaze_csv = neon_timeseries_dir + "/gaze.csv"

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
            logging.FileHandler(Path(output_dir, "whole_pipeline.log")),
            stream_handler,
        ],
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        f"Results will be saved in {output_dir} unless specified otherwise by mssg"
    )

    # Step 1: Calculate optic flow
    action_of_path, neon_of_path = calc_optic(
        neon_video=neon_vid_path,
        action_video=action_vid_path,
        output_dir=output_dir,
        of_choice=optic_flow_choice,
    )
    logger.info("Optic flow for both videos calculated")

    # Step 2: Estimate time offset and create action world timestamps
    action_result = pd.read_csv(action_of_path)
    neon_result = pd.read_csv(neon_of_path)
    align_videos(action_result, neon_result, action_vid_path, neon_timestamps)
    # check if world_timestamps.csv is created
    action_timestamps = neon_timeseries_dir + "/action_camera_timestamps.csv"
    if not Path(action_timestamps).exists():
        logger.error(f"{action_timestamps} not created!")
        raise FileNotFoundError(f"{action_timestamps} not created!")
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
        thresholds=mapper_thresholds,
    )
    logger.info(f"Gaze mapped to action video: {action_gaze_csv}")

    # Step 4 (Optional): Render simultaneous videos with gaze in both
    if render_video:
        video_path = Path(
            output_dir,
            f"video_render/Neon_Action_{image_matcher['choice']}_{optic_flow_choice}.mp4",
        )
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        save_comparison_video(
            action_video_path=action_vid_path,
            action_worldtimestamps_path=action_timestamps,
            action_gaze_paths_dict={image_matcher["choice"].upper(): action_gaze_csv},
            neon_video_path=neon_vid_path,
            neon_worldtimestamps_path=neon_timestamps,
            neon_gaze_path=neon_gaze_csv,
            save_video_path=video_path,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running whole pipeline for action camera gaze mapping"
    )
    parser.add_argument("--action_vid_path", type=str, help="Action video path.")
    parser.add_argument(
        "--neon_timeseries_path", type=str, help="Neon timeseries path."
    )
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument(
        "--matcher",
        choices=["loftr", "eloftr", "lg+disk"],
        default="eloftr",
        help="Image matcher to use.",
    )
    parser.add_argument(
        "--optic_flow_choice", choices=["lk", "farneback"], default="lk"
    )
    parser.add_argument(
        "--refresh_time_thrshld",
        type=float,
        help="Refresh time threshold.",
        default=None,
    )
    parser.add_argument(
        "--optic_flow_thrshld",
        type=int,
        help="Optic Flow threshold in deg.",
        default=None,
    )
    parser.add_argument(
        "--gaze_change_thrshld",
        type=int,
        help="Gaze change threshold in deg.",
        default=None,
    )
    parser.add_argument(
        "--render_video",
        type=bool,
        help="Render video with gaze overlay.",
        default=False,
    )

    args = parser.parse_args()

    param_lg = {"num_features": 2048, "gpu_num": 0}
    param_loftr = {"location": "indoor", "gpu_num": 0}
    param_eloftr = {"model_type": "opt", "gpu_num": 0}
    image_matcher_loftr = {"choice": "loftr", "parameters": param_loftr}
    image_matcher_lg = {"choice": "disk_lightglue", "parameters": param_lg}
    image_matcher_eloftr = {"choice": "efficient_loftr", "parameters": param_eloftr}

    image_matchers = {
        "loftr": image_matcher_loftr,
        "lg+disk": image_matcher_lg,
        "eloftr": image_matcher_eloftr,
    }
    main(
        action_vid_path=args.action_vid_path,
        neon_timeseries_dir=args.neon_timeseries_path,
        output_dir=args.output_dir,
        image_matcher=image_matchers[args.matcher],
        optic_flow_choice=args.optic_flow_choice,
        render_video=args.render_video,
        mapper_thresholds={
            "refresh_time_thrshld": args.refresh_time_thrshld,
            "optic_flow_thrshld": args.optic_flow_thrshld,
            "gaze_change_thrshld": args.gaze_change_thrshld,
        },
    )

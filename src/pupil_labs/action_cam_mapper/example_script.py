import os
import sys
import pandas as pd
import numpy as np
import logging
from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback
from utils import VideoHandler, write_timestamp_csv
from sync_videos import OffsetCalculator
from gaze_mapper import EgocentricMapper
from video_renderer import save_comparison_video
from pathlib import Path
import cProfile as profile
import argparse


def get_file(folder_path, file_suffix=".mp4", required_in_name="0"):
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder_path)
        for name in files
        if name.endswith(file_suffix) and required_in_name in name
    ][0]


def calc_optic_flow(
    neon_video, alternative_video, output_dir, optical_flow_method="farneback"
):
    output_dir = Path(output_dir, "optic_flow")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if optical_flow_method.lower() == "farneback":
        alternative_of = OpticFlowCalculatorFarneback(video_dir=alternative_video)
        neon_of = OpticFlowCalculatorFarneback(video_dir=neon_video)

    elif optical_flow_method.lower() == "lk":
        alternative_of = OpticFlowCalculatorLK(video_dir=alternative_video)
        neon_of = OpticFlowCalculatorLK(video_dir=neon_video)
    else:
        raise ValueError('Invalid optic flow choice. Choose from "farneback" or "lk"')

    optic_flow_neon = neon_of.process_video()
    optic_flow_alternative = alternative_of.process_video()

    alternative_saving_path = Path(
        output_dir, f"alternative_{optical_flow_method}_of.csv"
    )
    neon_saving_path = Path(output_dir, f"neon_{optical_flow_method}_of.csv")
    alternative_of.write_to_csv(output_file=alternative_saving_path)
    neon_of.write_to_csv(output_file=neon_saving_path)

    return alternative_saving_path, neon_saving_path


def align_videos(
    alternative_result,
    neon_result,
    alternative_vid_path,
    neon_timeseries_path,
    output_dir=None,
):
    offset_calc = OffsetCalculator(
        src=alternative_result["dy"].values,
        src_timestamps=alternative_result["start"].values,
        dst=neon_result["dy"].values,
        dst_timestamps=neon_result["start"].values,
        resampling_frequency=500,
    )
    t_offset, pearson_corr = offset_calc.estimate_time_offset()
    print(
        f"Estimated offset of alternative egocentric video with respect to Neon scene video: {t_offset} seconds (Pearson correlation: {pearson_corr})"
    )
    write_timestamp_csv(
        neon_timeseries_path,
        VideoHandler(alternative_vid_path).timestamps + t_offset,
        saving_path=output_dir,
    )


def main_mapper(
    alternative_vid_path,
    neon_vid_path,
    neon_timestamps,
    alternative_timestamps,
    neon_gaze_csv,
    neon_opticflow_csv,
    alternative_opticflow_csv,
    output_dir,
    matcher,
    optic_flow_choice,
    thresholds={
        "refresh_time_thrshld": None,
        "optic_flow_thrshld": None,
        "gaze_change_thrshld": None,
    },
    verbose=False,
):
    output_dir = Path(output_dir, f'mapped_gaze/{matcher["choice"].lower()}')

    param = matcher["parameters"]
    print(f'Using {matcher["choice"]} with parameters: {param}')
    mapper = EgocentricMapper(
        neon_gaze_csv=neon_gaze_csv,
        neon_video_path=neon_vid_path,
        alternative_video_path=alternative_vid_path,
        neon_timestamps=neon_timestamps,
        alternative_timestamps=alternative_timestamps,
        image_matcher=matcher["choice"],
        image_matcher_parameters=param,
        neon_opticflow_csv=neon_opticflow_csv,
        alternative_opticflow_csv=alternative_opticflow_csv,
        output_dir=output_dir,
        patch_size=1000,
        verbose=verbose,
    )
    gaze_csv_path = mapper.map_gaze(**thresholds)
    return gaze_csv_path


def main(
    alternative_vid_path,
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
    alternative_of_path, neon_of_path = calc_optic_flow(
        neon_video=neon_vid_path,
        alternative_video=alternative_vid_path,
        output_dir=output_dir,
        optical_flow_method=optic_flow_choice,
    )
    logger.info("Optic flow for both videos calculated")

    # Step 2: Estimate time offset and create action world timestamps
    alternative_result = pd.read_csv(alternative_of_path)
    neon_result = pd.read_csv(neon_of_path)
    align_videos(
        alternative_result,
        neon_result,
        alternative_vid_path,
        neon_timeseries_dir,
        output_dir,
    )
    # check if world_timestamps.csv is created
    alternative_camera_timestamps = output_dir + "/alternative_camera_timestamps.csv"
    if not Path(alternative_camera_timestamps).exists():
        logger.error(f"{alternative_camera_timestamps} not created!")
        raise FileNotFoundError(f"{alternative_camera_timestamps} not created!")
    # Step 3: Map gaze
    alternative_gaze_csv = main_mapper(
        alternative_vid_path=alternative_vid_path,
        neon_vid_path=neon_vid_path,
        neon_timestamps=neon_timestamps,
        alternative_timestamps=alternative_camera_timestamps,
        neon_gaze_csv=neon_gaze_csv,
        neon_opticflow_csv=neon_of_path,
        alternative_opticflow_csv=alternative_of_path,
        output_dir=output_dir,
        matcher=image_matcher,
        optic_flow_choice=optic_flow_choice,
        thresholds=mapper_thresholds,
    )
    logger.info(f"Gaze mapped to action video: {alternative_gaze_csv}")

    # Step 4 (Optional): Render simultaneous videos with gaze in both
    if render_video:
        video_path = Path(
            output_dir,
            f"video_render/Neon_Action_{image_matcher['choice']}_{optic_flow_choice}.mp4",
        )
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        save_comparison_video(
            alternative_video_path=alternative_vid_path,
            alternative_timestamps_path=alternative_camera_timestamps,
            alternative_gaze_paths_dict={"Alternative video": alternative_gaze_csv},
            neon_video_path=neon_vid_path,
            neon_worldtimestamps_path=neon_timestamps,
            neon_gaze_path=neon_gaze_csv,
            save_video_path=video_path,
        )


def profiling_map(
    alternative_vid_path,
    neon_timeseries_dir,
    output_dir,
    image_matcher,
    optic_flow_choice="lk",
    render_video=False,
):

    profile_stats_path = str(Path(output_dir, "profile_stats"))
    profile.runctx(
        "main(alternative_vid_path=alternative_vid_path,neon_timeseries_dir=neon_timeseries_dir,output_dir=output_dir,image_matcher=image_matcher, optic_flow_choice=optic_flow_choice,render_video=render_video)",
        globals(),
        locals(),
        profile_stats_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running whole pipeline for alternative egocentric camera gaze mapping"
    )
    parser.add_argument(
        "--alternative_vid_path", type=str, help="ALternative egocentric video path."
    )
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

    # profiling_map(
    #     alternative_vid_path=args.alternative_vid_path,
    #     neon_timeseries_dir=args.neon_timeseries_path,
    #     output_dir=args.output_dir,
    #     image_matcher=image_matchers[args.matcher],
    #     optic_flow_choice=args.optic_flow_choice,
    #     render_video=args.render_video,

    # )
    main(
        alternative_vid_path=args.alternative_vid_path,
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

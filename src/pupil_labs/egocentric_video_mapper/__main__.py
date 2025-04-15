import argparse
import json
import logging
import sys
from pathlib import Path

from pupil_labs.egocentric_video_mapper.gaze_mapper import (
    EgocentricFixationMapper,
    EgocentricMapper,
)
from pupil_labs.egocentric_video_mapper.optic_flow import calculate_optic_flow
from pupil_labs.egocentric_video_mapper.sync_videos import OffsetCalculator
from pupil_labs.egocentric_video_mapper.utils import (
    generate_comparison_video_kwargs,
    generate_mapper_kwargs,
    write_timestamp_csv,
)
from pupil_labs.egocentric_video_mapper.video_handler import VideoHandler
from pupil_labs.egocentric_video_mapper.video_renderer import (
    save_comparison_video,
    save_gaze_video,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str_lower(s):
    return s.lower()


def align_videos(
    alternative_result,
    neon_result,
    alternative_vid_path,
    neon_timeseries_dir,
    output_dir=None,
):
    offset_calc = OffsetCalculator(
        src=alternative_result["dy"].values,
        src_timestamps_sec=alternative_result["start"].values,
        dst=neon_result["dy"].values,
        dst_timestamps_sec=neon_result["start"].values,
        resampling_frequency=500,
    )
    t_offset, pearson_corr = offset_calc.estimate_time_offset()
    print(
        f"Estimated offset of alternative egocentric video with respect to Neon scene video: {t_offset} seconds (Pearson correlation: {pearson_corr})"
    )
    write_timestamp_csv(
        neon_timeseries_dir,
        VideoHandler(alternative_vid_path).timestamps + t_offset,
        output_file_dir=output_dir,
    )


def init_parser():
    parser = argparse.ArgumentParser(description="Pupil Labs - Egocentric Video Mapper")

    parser.add_argument(
        "--neon_timeseries_dir",
        type=str,
        help="Path to the uncompressed Neon 'Timeseries Data + Scene Video' directory.",
    )

    parser.add_argument(
        "--alternative_vid_path",
        type=str,
        help="Alternative egocentric video path.",
    )

    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument(
        "--mapping_choice",
        type=str_lower,
        choices=["fixations", "gaze", "both"],
        default="both",
        help="Mapping type.",
    )
    parser.add_argument(
        "--optic_flow_choice",
        type=str_lower,
        choices=["lucas-kanade", "farneback"],
        default="lucas-kanade",
    )

    parser.add_argument(
        "--matcher",
        choices=["Efficient_LOFTR", "LOFTR", "DISK_LightGlue", "DeDoDe_LightGlue"],
        default="Efficient_LOFTR",
        help="Image matcher to use in Egocentric Mapper.",
    )

    parser.add_argument(
        "--refresh_time_thrshld",
        type=float,
        help="Refresh time threshold.",
        default=None,
    )
    parser.add_argument(
        "--optic_flow_thrshld",
        type=float,
        help="Optic Flow threshold in deg.",
        default=None,
    )
    parser.add_argument(
        "--gaze_change_thrshld",
        type=float,
        help="Gaze change threshold in deg.",
        default=None,
    )

    parser.add_argument(
        "--render_comparison_video",
        type=str2bool,
        help="Render video comparing Neon Scene and alternative camera with gaze overlays.",
        default=False,
    )

    parser.add_argument(
        "--render_video",
        type=str2bool,
        help="Render video from alternative camera with gaze overlay.",
        default=False,
    )
    parser.add_argument(
        "--logging_level_file",
        default="INFO",
        help="Logging level for saving to log file",
    )

    return parser


def check_and_correct_args(args):
    try:
        args.gaze_change_thrshld = (
            None if args.gaze_change_thrshld == 0 else args.gaze_change_thrshld
        )
    except AttributeError:
        args.gaze_change_thrshld = None
    try:
        args.refresh_time_thrshld = (
            None if args.refresh_time_thrshld == 0 else args.refresh_time_thrshld
        )
    except AttributeError:
        args.refresh_time_thrshld = None
    try:
        args.optic_flow_thrshld = (
            None if args.optic_flow_thrshld == 0 else args.optic_flow_thrshld
        )
    except AttributeError:
        args.optic_flow_thrshld = None
    if not hasattr(args, "logging_level_file"):
        args.logging_level_file = "ERROR"
    args.mapping_choice = args.mapping_choice.lower()
    return args


def main(args=None):
    if args is None:
        parser = init_parser()
        args = parser.parse_args()
    args = check_and_correct_args(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir, "egocentric_video_mapper_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=Path(args.output_dir, f"egocentric_mapper_pipeline.log"),
        filemode="a",
        format="[%(levelname)s]  %(funcName)s function in %(name)s (%(asctime)s):  %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=args.logging_level_file,
    )

    logger.info(
        "◎ Egocentric Mapper Module by Pupil Labs",
    )
    mssg_mapping_choice = (
        "both fixations and gaze"
        if args.mapping_choice == "both"
        else args.mapping_choice
    )
    logger.info(
        f"Mapping {mssg_mapping_choice} into Alternative Egocentric View.\nResults will be saved in {args.output_dir} unless specified otherwise by message."
    )

    neon_of_result, alternative_of_result = calculate_optic_flow(
        neon_timeseries_dir=args.neon_timeseries_dir,
        alternative_video_path=args.alternative_vid_path,
        output_dir=args.output_dir,
        optic_flow_method=args.optic_flow_choice,
    )
    logger.info("Optic flow for both videos calculated")

    align_videos(
        alternative_result=alternative_of_result,
        neon_result=neon_of_result,
        alternative_vid_path=args.alternative_vid_path,
        neon_timeseries_dir=args.neon_timeseries_dir,
        output_dir=args.output_dir,
    )

    mapper_kwargs = generate_mapper_kwargs(
        neon_timeseries_dir=args.neon_timeseries_dir,
        alternative_vid_path=args.alternative_vid_path,
        output_dir=args.output_dir,
        matcher_choice=args.matcher,
        logging_level=args.logging_level_file,
    )

    if args.mapping_choice == "fixations" or args.mapping_choice == "both":
        mapper = EgocentricFixationMapper(**mapper_kwargs)
        fixations_csv_path = mapper.map_fixations()
        print(f"Fixations mapped to alternative camera saved at {fixations_csv_path}")

    if args.mapping_choice == "gaze" or args.mapping_choice == "both":
        mapper = EgocentricMapper(**mapper_kwargs)
        gaze_csv_path = mapper.map_gaze(
            refresh_time_thrshld=args.refresh_time_thrshld,
            optic_flow_thrshld=args.optic_flow_thrshld,
            gaze_change_thrshld=args.gaze_change_thrshld,
        )

        print(f"Gaze mapped to alternative camera video saved at {gaze_csv_path}")
        if args.render_comparison_video:
            comparison_kwargs = generate_comparison_video_kwargs(
                neon_timeseries_dir=args.neon_timeseries_dir,
                alternative_vid_path=args.alternative_vid_path,
                mapped_gaze_path=gaze_csv_path,
                output_dir=args.output_dir,
            )
            save_comparison_video(**comparison_kwargs)

        if args.render_video:
            gaze_video_args = {
                "video_path": args.alternative_vid_path,
                "timestamps_path": Path(
                    args.output_dir, "alternative_camera_timestamps.csv"
                ),
                "gaze_path": Path(gaze_csv_path),
                "save_video_path": Path(
                    args.output_dir,
                    "alternative_camera_gaze_overlay.mp4",
                ),
            }
            save_gaze_video(**gaze_video_args)


if __name__ == "__main__":
    # python -m pupil_labs.egocentric_video_mapper --neon_timeseries_dir 'Path/To/NeonTimeSeriesFolder' --alternative_vid_path 'Path/To/AlternativeVideo.ext' --output_dir "/Path/To/OutputFolder" --mapping_choice 'fixations'
    ## python -m pupil_labs.egocentric_video_mapper --neon_timeseries_dir 'Path/To/NeonTimeSeriesFolder' --alternative_vid_path 'Path/To/AlternativeVideo.ext' --output_dir "/Path/To/OutputFolder" --mapping_choice 'gaze' --render_comparison_video True --render_video True
    main()

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from pupil_labs.egocentric_video_mapper.gaze_mapper import EgocentricMapper
from pupil_labs.egocentric_video_mapper.optic_flow import calculate_optic_flow
from pupil_labs.egocentric_video_mapper.sync_videos import OffsetCalculator
from pupil_labs.egocentric_video_mapper.utils import (
    generate_comparison_video_kwargs,
    generate_mapper_kwargs,
    write_timestamp_csv,
)

from .video_handler import VideoHandler
from .video_renderer import save_comparison_video, save_gaze_video


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


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Running whole pipeline for alternative egocentric camera gaze mapping"
        )

        parser.add_argument(
            "--neon_timeseries_dir",
            type=Path,
            help="Path to the uncompressed Neon 'Timeseries Data + Scene Video' directory.",
        )

        parser.add_argument(
            "--alternative_vid_path",
            type=Path,
            help="Alternative egocentric video path.",
        )

        parser.add_argument("--output_dir", type=Path, help="Output directory.")

        parser.add_argument(
            "--optic_flow_choice",
            choices=["Lucas-Kanade", "Farneback"],
            default="Lucas-Kanade",
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
            type=bool,
            help="Render video comparing Neon Scene and alternative camera with gaze overlays.",
            default=False,
        )

        parser.add_argument(
            "--render_video",
            type=bool,
            help="Render video from alternative camera with gaze overlay.",
            default=False,
        )

        args = parser.parse_args()
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(funcName)s function in %(name)s: %(message)s"
        )
    )
    # stream_handler = RichHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(
    #     logging.Formatter("%(funcName)s in %(name)s: %(message)s")
    # )
    logging.basicConfig(
        format="[%(levelname)s]  %(funcName)s function in %(name)s (%(asctime)s):  %(message)s",
        handlers=[
            logging.FileHandler(
                Path(args.output_dir, "egocentric_mapper_pipeline.log")
            ),
            stream_handler,
        ],
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "[white bold on #0d122a]◎ Egocentric Mapper Module by Pupil Labs[/]",
        extra={"markup": True},
    )
    logger.info(
        f"Results will be saved in {args.output_dir} unless specified otherwise by message."
    )
    logger.info(f"Arguments: {args}")

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
        optic_flow_method=args.optic_flow_choice,
    )
    stream_handler.setLevel(logging.WARNING)
    mapper = EgocentricMapper(**mapper_kwargs, logging_level="INFO")
    gaze_csv_path = mapper.map_gaze(
        refresh_time_thrshld=args.refresh_time_thrshld,
        optic_flow_thrshld=args.optic_flow_thrshld,
        gaze_change_thrshld=args.gaze_change_thrshld,
    )

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
                f"rendered_videos/{args.matcher.lower()}_{args.optic_flow_choice.lower()}.mp4",
            ),
        }
        save_gaze_video(**gaze_video_args)


if __name__ == "__main__":
    # minimal example
    # python -m pupil_labs.action_cam_mapper --neon_timeseries_dir /users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62 --alternative_vid_path /users/sof/video_examples/second_video/20240523_171941_000.mp4 --output_dir /users/sof/action_map_experiments/output5 --optic_flow_choice Lucas-Kanade --matcher Efficient_LOFTR --refresh_time_thrshld 0.5 --optic_flow_thrshld None --gaze_change_thrshld 0 --render_comparison_video False --render_video False
    main()
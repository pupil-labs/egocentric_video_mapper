import sys
import logging
from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback
from utils import (
    VideoHandler,
    write_timestamp_csv,
    generate_mapper_kwargs,
    generate_comparison_video_kwargs,
)
from sync_videos import OffsetCalculator
from gaze_mapper import EgocentricMapper
from video_renderer import save_comparison_video, save_gaze_video
from pathlib import Path
import argparse


def calculate_optic_flow(
    neon_timeseries_dir,
    alternative_video_path,
    output_dir,
    optical_flow_method="farneback",
):
    neon_video_path = Path(neon_timeseries_dir).rglob("*.mp4").__next__()
    optical_flow_method = optical_flow_method.lower()
    output_dir = Path(output_dir, "optic_flow")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if optical_flow_method.lower() == "farneback":
        alternative_of = OpticFlowCalculatorFarneback(video_path=alternative_video_path)
        neon_of = OpticFlowCalculatorFarneback(video_path=neon_video_path)
        method = "farneback"

    elif optical_flow_method.lower() == "lucas-kanade":
        alternative_of = OpticFlowCalculatorLK(video_path=alternative_video_path)
        neon_of = OpticFlowCalculatorLK(video_path=neon_video_path)
        method = "lk"
    else:
        raise ValueError(
            'Invalid optic flow choice. Choose from "farneback" or "lucas-kanade"'
        )

    optic_flow_neon = neon_of.process_video(
        output_file_path=Path(output_dir, f"neon_{method}_of.csv")
    )
    optic_flow_alternative = alternative_of.process_video(
        output_file_path=Path(output_dir, f"alternative_{method}_of.csv")
    )

    return optic_flow_neon, optic_flow_alternative


def align_videos(
    alternative_result,
    neon_result,
    alternative_vid_path,
    neon_timeseries_dir,
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
            logging.FileHandler(Path(args.output_dir, "whole_pipeline.log")),
            stream_handler,
        ],
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    logging.info("Logging setup complete. This is a test log message.")
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
        optical_flow_method=args.optic_flow_choice,
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
        optical_flow_method=args.optic_flow_choice,
    )
    mapper = EgocentricMapper(**mapper_kwargs)
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
        method = (
            "lk" if args.optic_flow_choice.lower() == "lucas-kanade" else "farneback"
        )
        gaze_video_args = {
            "video_path": args.alternative_vid_path,
            "timestamps_path": Path(
                args.output_dir, "alternative_camera_timestamps.csv"
            ),
            "gaze_path": Path(gaze_csv_path),
            "save_video_path": Path(
                args.output_dir, f"rendered_videos/{args.matcher.lower()}_{method}.mp4"
            ),
        }
        save_gaze_video(**gaze_video_args)


if __name__ == "__main__":
    # minimal example
    # python -m pupil_labs.action_cam_mapper --neon_timeseries_dir /home/rohit/Downloads/2022_02_24_15_00_00 --alternative_vid_path /home/rohit/Downloads/2022_02_24_15_00_00/2022_02_24_15_00_00.mp4 --output_dir /home/rohit/Downloads/2022_02_24_15_00_00/output --optic_flow_choice Lucas-Kanade --matcher Efficient_LOFTR --refresh_time_thrshld 0.5 --optic_flow_thrshld 0.5 --gaze_change_thrshld 0.5 --render_comparison_video True --render_video True
    main()

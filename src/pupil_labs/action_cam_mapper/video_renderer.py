import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from utils import VideoHandler
import logging
from tqdm import tqdm


FONT_CHOICE = cv.FONT_HERSHEY_SIMPLEX


def get_gaze_per_frame(gaze_file, video_timestamps):
    """This function search for the gaze coordinates with the closest world timestamp to the video world timestamps and returns an array of coordinates for every frame in the video

    Args:
        gaze_file (str): Path to the gaze file associated to a video
        video_timestamps (str): Path to the world timestamps to the frames in a video

    Returns:
        (ndarray): A numpy array containing the x,y coordinates for every entry in the video_timestamps (shape: (n_frames, 2))
    """
    scene_timestamps = pd.read_csv(video_timestamps)
    gaze_timestamps = pd.read_csv(gaze_file)
    coords = pd.merge_asof(
        scene_timestamps, gaze_timestamps, on="timestamp [ns]", direction="nearest"
    )
    return coords[["gaze x [px]", "gaze y [px]"]].to_numpy()


def pad_images_height(image_1, image_2):
    """Given two images of different sizes, this function pads the smaller image with zeros so that both images have the same height

    Args:
        image_1 (ndarray): Numpy array containing an image with shape (height, width, channels)
        image_2 (ndarray): Numpy array containing an image with shape (height, width, channels)
    """
    height_1 = image_1.shape[0]
    height_2 = image_2.shape[0]

    pad_bottom = abs(height_1 - height_2)

    if height_1 > height_2:
        image_2 = np.pad(
            image_2,
            ((0, pad_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        image_1 = np.pad(
            image_1,
            ((0, pad_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return image_1, image_2


def write_text_on_frame(
    frame, text, position, font=FONT_CHOICE, color=(0, 0, 0), thickness=4, size=3
):
    return cv.putText(
        frame,
        text,
        position,
        font,
        size,
        color,
        thickness,
        cv.LINE_AA,
    )


def draw_gaze_on_frame(
    frame, gaze_coords, gaze_radius=20, gaze_thickness=4, gaze_color=(0, 0, 255)
):
    return cv.circle(
        frame,
        np.int32(gaze_coords),
        gaze_radius,
        gaze_color,
        gaze_thickness,
    )


def view_video(
    alternative_video_path,
    alternative_timestamps_path,
    alternative_gaze_paths_dict,
    neon_video_path,
    neon_worldtimestamps_path,
    neon_gaze_path,
):

    alternative_coords = {
        matcher: get_gaze_per_frame(
            gaze_file=path, video_timestamps=alternative_timestamps_path
        )
        for matcher, path in alternative_gaze_paths_dict.items()
    }
    neon_gaze_coords_list = get_gaze_per_frame(
        gaze_file=neon_gaze_path, video_timestamps=neon_worldtimestamps_path
    )

    alternative_video = VideoHandler(alternative_video_path)
    neon_video = VideoHandler(neon_video_path)

    neon_timestamps = pd.read_csv(
        neon_worldtimestamps_path, dtype={"timestamp [ns]": np.float64}
    )
    alternative_timestamps = pd.read_csv(
        alternative_timestamps_path, dtype={"timestamp [ns]": np.float64}
    )

    alternative_time = alternative_timestamps["timestamp [ns]"].values
    alternative_time -= neon_timestamps["timestamp [ns]"].values[0]
    alternative_time /= 1e9

    neon_gaze_dict = {
        t: gaze for t, gaze in zip(neon_video.timestamps, neon_gaze_coords_list)
    }
    alternative_gaze_dict = {
        matcher: {t: gaze for t, gaze in zip(alternative_time, coords_list)}
        for matcher, coords_list in alternative_coords.items()
    }

    video_height = max(alternative_video.height, neon_video.height)
    video_width = neon_video.width + alternative_video.width * len(
        alternative_gaze_dict.keys()
    )
    video_height = video_height // len(alternative_gaze_dict.keys())
    video_width = video_width // len(alternative_gaze_dict.keys())

    for i, t in enumerate(alternative_time):
        neon_frame = neon_video.get_frame_by_timestamp(t)
        neon_frame = cv.cvtColor(neon_frame, cv.COLOR_BGR2RGB)

        neon_frame_gaze = draw_gaze_on_frame(
            neon_frame.copy(), neon_gaze_dict[neon_video.get_closest_timestamp(t)[0]]
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze, "Neon Scene Camera", (50, 80), thickness=5
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze, "Neon Scene Camera", (50, 80), color=(255, 255, 255)
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            f"Time: {neon_video.get_closest_timestamp(t)[0]:.3f}",
            (50, 150),
            size=2,
            thickness=5,
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            f"Time: {neon_video.get_closest_timestamp(t)[0]:.3f}",
            (50, 150),
            size=2,
            color=(255, 255, 255),
        )

        all_frames = neon_frame_gaze.copy()

        alternative_frame = alternative_video.get_frame_by_timestamp(
            alternative_video.timestamps[i]
        )
        alternative_frame = cv.cvtColor(alternative_frame, cv.COLOR_RGB2BGR)

        for matcher in alternative_gaze_dict.keys():
            gaze = alternative_gaze_dict[matcher][t]
            alternative_frame_gaze = draw_gaze_on_frame(
                alternative_frame.copy(),
                gaze,
            )
            alternative_frame_gaze = write_text_on_frame(
                alternative_frame_gaze, matcher, (50, 80), thickness=5
            )
            alternative_frame_gaze = write_text_on_frame(
                alternative_frame_gaze, matcher, (50, 80), color=(255, 255, 255)
            )
            alternative_frame_gaze = write_text_on_frame(
                alternative_frame_gaze, f"Time: {t:.3f}", (50, 150), size=2, thickness=5
            )
            alternative_frame_gaze = write_text_on_frame(
                alternative_frame_gaze,
                f"Time: {t:.3f}",
                (50, 150),
                size=2,
                color=(255, 255, 255),
            )

            all_frames, alternative_frame_gaze = pad_images_height(
                all_frames, alternative_frame_gaze
            )
            all_frames = np.concatenate([all_frames, alternative_frame_gaze], axis=1)

        all_frames = cv.resize(all_frames, (video_width, video_height))
        cv.imshow("Comparing mappings", all_frames)
        cv.waitKey(100)
        pressedKey = cv.waitKey(50) & 0xFF
        if pressedKey == ord(" "):
            print("Paused")
            cv.waitKey(0)
        if pressedKey == ord("q"):
            break
    cv.destroyAllWindows()


def save_gaze_video(video_path, timestamps_path, gaze_path, save_video_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    gaze_coordinates = get_gaze_per_frame(
        gaze_file=gaze_path, video_timestamps=timestamps_path
    )
    video = VideoHandler(video_path)

    video_height = video.height
    video_width = video.width
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
    gaze_video = cv.VideoWriter(
        str(save_video_path), fourcc, int(video.fps), (video_width, video_height)
    )
    logger.info(f"Saving video at {save_video_path}")
    logger.info(f"Video width: {video_width}, Video height: {video_height}")

    for i, gaze in enumerate(tqdm(gaze_coordinates)):
        frame = video.get_frame_by_timestamp(video.timestamps[i])
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = draw_gaze_on_frame(frame, gaze)
        gaze_video.write(frame.astype(np.uint8))
    gaze_video.release()
    logger.info(f"Video saved at {save_video_path}")
    print(f"Video saved at {save_video_path}")


def save_comparison_video(
    alternative_video_path,
    alternative_timestamps_path,
    alternative_gaze_paths_dict,
    neon_video_path,
    neon_worldtimestamps_path,
    neon_gaze_path,
    save_video_path,
    same_frame=False,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    alternative_coords = {
        matcher: get_gaze_per_frame(
            gaze_file=path, video_timestamps=alternative_timestamps_path
        )
        for matcher, path in alternative_gaze_paths_dict.items()
    }
    neon_gaze_coords_list = get_gaze_per_frame(
        gaze_file=neon_gaze_path, video_timestamps=neon_worldtimestamps_path
    )

    alternative_video = VideoHandler(alternative_video_path)
    neon_video = VideoHandler(neon_video_path)

    neon_timestamps = pd.read_csv(
        neon_worldtimestamps_path, dtype={"timestamp [ns]": np.float64}
    )
    alternative_timestamps = pd.read_csv(
        alternative_timestamps_path, dtype={"timestamp [ns]": np.float64}
    )

    alternative_time = alternative_timestamps["timestamp [ns]"].values
    alternative_time -= neon_timestamps["timestamp [ns]"].values[0]
    alternative_time /= 1e9
    neon_gaze_dict = {
        t: gaze for t, gaze in zip(neon_video.timestamps, neon_gaze_coords_list)
    }
    alternative_gaze_dict = {
        matcher: {t: gaze for t, gaze in zip(alternative_time, coords_list)}
        for matcher, coords_list in alternative_coords.items()
    }

    video_height = max(alternative_video.height, neon_video.height)
    video_width = neon_video.width + (
        alternative_video.width * len(alternative_gaze_dict.keys())
        if not same_frame
        else alternative_video.width
    )
    if not same_frame:
        video_height = video_height // len(alternative_gaze_dict.keys())
        video_width = video_width // len(alternative_gaze_dict.keys())

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
    video = cv.VideoWriter(
        str(save_video_path),
        fourcc,
        int(alternative_video.fps),
        (video_width, video_height),
    )
    logger.info(f"Saving video at {save_video_path}")
    logger.info(f"Video width: {video_width}, Video height: {video_height}")
    gaze_color_list = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
    ]

    for i, t in enumerate(tqdm(alternative_time)):
        neon_frame = neon_video.get_frame_by_timestamp(t)
        neon_frame = cv.cvtColor(neon_frame, cv.COLOR_BGR2RGB)
        neon_frame_gaze = draw_gaze_on_frame(
            neon_frame.copy(), neon_gaze_dict[neon_video.get_closest_timestamp(t)[0]]
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            "Neon Scene Camera",
            (50, 80),
            thickness=5,
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            "Neon Scene Camera",
            (50, 80),
            color=(255, 255, 255),
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            f"Time: {neon_video.get_closest_timestamp(t)[0]:.3f}s",
            (50, 150),
            size=2,
            thickness=5,
        )
        neon_frame_gaze = write_text_on_frame(
            neon_frame_gaze,
            f"Time: {neon_video.get_closest_timestamp(t)[0]:.3f}s",
            (50, 150),
            size=2,
            color=(255, 255, 255),
        )
        all_frames = neon_frame_gaze.copy()

        alternative_frame = alternative_video.get_frame_by_timestamp(
            alternative_video.timestamps[i]
        )
        alternative_frame = cv.cvtColor(alternative_frame, cv.COLOR_RGB2BGR)

        for i_matcher, matcher in enumerate(alternative_gaze_dict.keys()):
            gaze = alternative_gaze_dict[matcher][t]
            if same_frame:
                alternative_frame = draw_gaze_on_frame(
                    alternative_frame, gaze, gaze_color=gaze_color_list[i_matcher]
                )
                alternative_frame = write_text_on_frame(
                    alternative_frame,
                    matcher,
                    (50, 50 + 70 * i_matcher),
                    size=2,
                    color=gaze_color_list[i_matcher],
                )
            else:
                alternative_frame = write_text_on_frame(
                    alternative_frame,
                    f"Time: {t:.3f}s",
                    (50, 150),
                    size=2,
                    thickness=5,
                )
                alternative_frame = write_text_on_frame(
                    alternative_frame,
                    f"Time: {t:.3f}s",
                    (50, 150),
                    size=2,
                    color=(255, 255, 255),
                )
                alternative_frame_gaze = draw_gaze_on_frame(
                    alternative_frame.copy(), gaze
                )
                alternative_frame_gaze = write_text_on_frame(
                    alternative_frame_gaze,
                    matcher,
                    (50, 80),
                    thickness=5,
                )
                alternative_frame_gaze = write_text_on_frame(
                    alternative_frame_gaze,
                    matcher,
                    (50, 80),
                    color=(255, 255, 255),
                )
                all_frames, alternative_frame_gaze = pad_images_height(
                    all_frames, alternative_frame_gaze
                )
                all_frames = np.concatenate(
                    [all_frames, alternative_frame_gaze], axis=1
                )
        if same_frame:
            alternative_frame = write_text_on_frame(
                alternative_frame,
                f"Time: {t:.3f}s",
                (50, 50 + 70 * len(alternative_gaze_dict.keys())),
                size=2,
                thickness=5,
            )
            alternative_frame = write_text_on_frame(
                alternative_frame,
                f"Time: {t:.3f}s",
                (50, 50 + 70 * len(alternative_gaze_dict.keys())),
                size=2,
                color=(255, 255, 255),
            )
            all_frames, alternative_frame = pad_images_height(
                all_frames, alternative_frame
            )
            all_frames = np.concatenate([all_frames, alternative_frame], axis=1)
        all_frames = cv.resize(all_frames, (video_width, video_height))  # w,h
        video.write(all_frames.astype(np.uint8))
    video.release()
    logger.info(f"Video saved at {save_video_path}")
    print(f"Video saved at {save_video_path}")


if __name__ == "__main__":
    videoss = [
        "office1",
        # "street_very_inclined",
        # "street_slight_inclined",
    ]
    datset_choice = "mini_dataset"
    datset_exp = "minidataset_hoover"
    exp = "gz"
    if exp == "of":
        dir_exp = "opticalflow_ts"
    elif exp == "rf":
        dir_exp = "refreshtime_ts"
    else:
        dir_exp = "gaze_ts"

    for video_sel in videoss:

        neon_timestamps = list(
            Path(f"/users/sof/{datset_choice}/{video_sel}").rglob(
                "*/world_timestamps.csv"
            )
        )[0]
        neon_gaze_path = list(
            Path(f"/users/sof/{datset_choice}/{video_sel}").rglob("*/gaze.csv")
        )[0]
        neon_vid_path = list(neon_gaze_path.parent.rglob("*.mp4"))[0]

        action_vid_path = list(
            Path(f"/users/sof/{datset_choice}/{video_sel}").rglob("AVun*.mp4")
        )[0]
        action_timestamps = list(
            Path(f"/users/sof/{datset_choice}/{video_sel}").rglob(
                "*/action_camera_timestamps.csv"
            )
        )[0]

        gazes_dict = {}

        gazes_dict["Insta360 GO3 Camera"] = (
            f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/baseline/no_thresh/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
        )

        if exp == "rf":
            gazes_dict[f"Action Cam (ELOFTR {exp}0.5)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/0.5/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
            gazes_dict[f"Action Cam (ELOFTR {exp}0.25)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/0.25/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
            gazes_dict[f"Action Cam (ELOFTR {exp}0.05)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/0.05/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
        else:
            gazes_dict[f"Action Cam (ELOFTR {exp}1)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/1/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
            gazes_dict[f"Action Cam (ELOFTR {exp}5)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/5/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
            gazes_dict[f"Action Cam (ELOFTR {exp}10)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/10/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )
            gazes_dict[f"Action Cam (ELOFTR {exp}20)"] = (
                f"/users/sof/action_map_experiments/{datset_exp}/{video_sel}/{dir_exp}/20/mapped_gaze/efficient_loftr/action_gaze_lk.csv"
            )

        save_comparison_video(
            alternative_video_path=action_vid_path,
            alternative_timestamps_path=action_timestamps,
            alternative_gaze_paths_dict=gazes_dict,
            neon_video_path=neon_vid_path,
            neon_worldtimestamps_path=neon_timestamps,
            neon_gaze_path=neon_gaze_path,
            save_video_path=f"/users/sof/action_map_experiments/alpha_min_rendering/{video_sel}/same_Neon_Action_{exp}.mp4",
            same_frame=True,
        )

        # view_video(
        #     alternative_video_path=action_vid_path,
        #     alternative_timestamps_path=action_timestamps,
        #     alternative_gaze_paths_dict=gazes_dict,
        #     neon_video_path=neon_vid_path,
        #     neon_worldtimestamps_path=neon_timestamps,
        #     neon_gaze_path=neon_gaze_path,
        # )

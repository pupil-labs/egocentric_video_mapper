import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from utils import VideoHandler


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
    scene_ns = scene_timestamps['timestamp [ns]'].to_numpy()
    gaze_ns = gaze_timestamps['timestamp [ns]'].to_numpy()
    list_coords = []
    for scene_time in scene_ns:
        # matching scene timestamp to the smallest timestamp in gaze_ns that is greater than the scene timestamp, since gaze timestamping starts after world scene timestamping
        gaze_indexing = np.argmin(np.abs(gaze_ns-scene_time))
        coords = gaze_timestamps.iloc[gaze_indexing][['gaze x [px]', 'gaze y [px]']]
        list_coords.append(coords.to_numpy())
    return list_coords


def pad_images_height(image_1, image_2):
    """Given two images of different sizes, this function pads the smaller image with zeros so that both images have the same height

    Args:
        image_1 (ndarray): Numpy array containing an image with shape (height, width, channels)
        image_2 (ndarray): Numpy array containing an image with shape (height, width, channels)
    """
    height_1 = image_1.shape[0]
    height_2 = image_2.shape[0]

    pad_bottom = abs(height_1-height_2)

    if height_1 > height_2:
        image_2 = np.pad(image_2, ((0, pad_bottom), (0, 0), (0, 0)),
                         mode='constant', constant_values=0)
    else:
        image_1 = np.pad(image_1, ((0, pad_bottom), (0, 0), (0, 0)),
                         mode='constant', constant_values=0)
    return image_1, image_2


def view_video(action_video_path,
        action_worldtimestamps_path,
        action_gaze_paths_dict,
        neon_video_path,
        neon_worldtimestamps_path,
        neon_gaze_path):
    
    action_coords = {matcher:get_gaze_per_frame(gaze_file=path, video_timestamps=action_worldtimestamps_path) for matcher,path in action_gaze_paths_dict.items()}
    neon_gaze_coords_list = get_gaze_per_frame(
        gaze_file=neon_gaze_path, video_timestamps=neon_worldtimestamps_path)
    
    action_video = VideoHandler(action_video_path)
    neon_video = VideoHandler(neon_video_path)

    neon_timestamps = pd.read_csv(neon_worldtimestamps_path, dtype={
                                  'timestamp [ns]': np.float64})
    action_timestamps = pd.read_csv(action_worldtimestamps_path, dtype={'timestamp [ns]': np.float64})

    action_time = action_timestamps['timestamp [ns]'].values
    action_time -= neon_timestamps['timestamp [ns]'].values[0]
    action_time /= 1e9
    neon_gaze_dict = {t: gaze for t, gaze in zip(
        neon_video.timestamps, neon_gaze_coords_list)}
    action_gaze_dict ={matcher:{t:gaze for t,gaze in zip(action_time,coords_list)} for matcher,coords_list in action_coords.items()}
    
    video_height = max(action_video.height, neon_video.height)
    video_width = neon_video.width+action_video.width*len(action_gaze_dict.keys())
    video_height =video_height//len(action_gaze_dict.keys())
    video_width = video_width//len(action_gaze_dict.keys())
    
    gaze_radius = 20
    gaze_thickness = 4
    gaze_color = (0, 0, 255)

    for i, t in enumerate(action_time):
        neon_frame = neon_video.get_frame_by_timestamp(t)
        neon_frame = cv.cvtColor(neon_frame, cv.COLOR_BGR2RGB)

        neon_frame_gaze=cv.circle(neon_frame.copy(), np.int32(
            neon_gaze_dict[neon_video.get_closest_timestamp(t)]), gaze_radius, gaze_color, gaze_thickness)
        cv.putText(neon_frame_gaze, f'Neon Scene', (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(neon_frame_gaze, f'Time: {neon_video.get_closest_timestamp(t)}',
                   (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        
        all_frames = neon_frame_gaze.copy()

        action_frame = action_video.get_frame_by_timestamp(action_video.timestamps[i])
        action_frame = cv.cvtColor(action_frame, cv.COLOR_RGB2BGR)

        for matcher in action_gaze_dict.keys():
            gaze = action_gaze_dict[matcher][t]
            action_frame_gaze=cv.circle(action_frame.copy(), np.int32(gaze), gaze_radius, gaze_color, gaze_thickness)
            cv.putText(action_frame_gaze, f'Action Cam ({matcher})', (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(action_frame_gaze, f'Time: {t}', (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            all_frames, action_frame_gaze = pad_images_height(all_frames, action_frame_gaze)
            all_frames =np.concatenate([all_frames, action_frame_gaze], axis=1)

        all_frames = cv.resize(all_frames, (video_width,video_height)) #w,h
        cv.imshow('both_frames', all_frames)
        cv.waitKey(100)
        pressedKey = cv.waitKey(50) & 0xFF
        if pressedKey == ord(' '):
            print('Paused')
            cv.waitKey(0)
        if pressedKey == ord('q'):
            break
    cv.destroyAllWindows()



def save_video(action_video_path,
        action_worldtimestamps_path,
        action_gaze_paths_dict,
        neon_video_path,
        neon_worldtimestamps_path,
        neon_gaze_path,
        save_video_path):
    
    action_coords = {matcher:get_gaze_per_frame(gaze_file=path, video_timestamps=action_worldtimestamps_path) for matcher,path in action_gaze_paths_dict.items()}
    neon_gaze_coords_list = get_gaze_per_frame(
        gaze_file=neon_gaze_path, video_timestamps=neon_worldtimestamps_path)
    
    action_video = VideoHandler(action_video_path)
    neon_video = VideoHandler(neon_video_path)

    neon_timestamps = pd.read_csv(neon_worldtimestamps_path, dtype={
                                  'timestamp [ns]': np.float64})
    action_timestamps = pd.read_csv(action_worldtimestamps_path, dtype={'timestamp [ns]': np.float64})

    action_time = action_timestamps['timestamp [ns]'].values
    action_time -= neon_timestamps['timestamp [ns]'].values[0]
    action_time /= 1e9
    neon_gaze_dict = {t: gaze for t, gaze in zip(
        neon_video.timestamps, neon_gaze_coords_list)}
    action_gaze_dict ={matcher:{t:gaze for t,gaze in zip(action_time,coords_list)} for matcher,coords_list in action_coords.items()}
    

    video_height = max(action_video.height, neon_video.height)
    video_width = neon_video.width+action_video.width*len(action_gaze_dict.keys())
    video_height =video_height//len(action_gaze_dict.keys())
    video_width = video_width//len(action_gaze_dict.keys())
    
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    print(video_width, video_height)
    video = cv.VideoWriter(str(save_video_path), fourcc, int(
        action_video.fps), (video_width,video_height))
    print(f'Saving video at {save_video_path}')

    gaze_radius = 20
    gaze_thickness = 4
    gaze_color = (0, 0, 255)

    for i, t in enumerate(action_time):
        neon_frame = neon_video.get_frame_by_timestamp(t)
        neon_frame = cv.cvtColor(neon_frame, cv.COLOR_BGR2RGB)

        neon_frame_gaze=cv.circle(neon_frame.copy(), np.int32(
            neon_gaze_dict[neon_video.get_closest_timestamp(t)[0]]), gaze_radius, gaze_color, gaze_thickness)
        cv.putText(neon_frame_gaze, f'Neon Scene', (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(neon_frame_gaze, f'Time: {neon_video.get_closest_timestamp(t)}',
                   (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        
        all_frames = neon_frame_gaze.copy()

        action_frame = action_video.get_frame_by_timestamp(action_video.timestamps[i])
        action_frame = cv.cvtColor(action_frame, cv.COLOR_RGB2BGR)

        for matcher in action_gaze_dict.keys():
            gaze = action_gaze_dict[matcher][t]
            action_frame_gaze=cv.circle(action_frame.copy(), np.int32(gaze), gaze_radius, gaze_color, gaze_thickness)
            cv.putText(action_frame_gaze, f'Action Cam ({matcher})', (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(action_frame_gaze, f'Time: {t}', (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            all_frames, action_frame_gaze = pad_images_height(all_frames, action_frame_gaze)
            all_frames =np.concatenate([all_frames, action_frame_gaze], axis=1)
        
        all_frames = cv.resize(all_frames, (video_width,video_height)) #w,h
        video.write(all_frames.astype(np.uint8))
    
    video.release()
    print('Finished')


if __name__ == "__main__":
    
    neon_vid_path='/users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62/e69049ea_0.0-42.986.mp4'
    neon_timestamps='/users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62/world_timestamps.csv'
    neon_gaze_path='/users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62/gaze.csv'

    action_vid_path='/users/sof/video_examples/second_video/20240523_171941_000.mp4'
    action_timestamps='/users/sof/video_examples/second_video/2024-05-23_16-47-35-a666ea62/action_camera_timestamps.csv'
    action_gaze_path_lg = '/users/sof/action_map_experiments/second_video_2/mapped_gaze/disk_lightglue/action_gaze_lnk.csv'
    action_gaze_path_loftr = '/users/sof/action_map_experiments/second_video_2/mapped_gaze/loftr/action_gaze_lnk.csv'

    save_video(action_video_path=action_vid_path,
        action_worldtimestamps_path=action_timestamps,
        action_gaze_paths_dict={'LOFTR':action_gaze_path_loftr, 'LG+DISK':action_gaze_path_lg},
        neon_video_path=neon_vid_path,
        neon_worldtimestamps_path=neon_timestamps,
        neon_gaze_path=neon_gaze_path,
        save_video_path='/users/sof/action_map_experiments/second_video_2/video_render/Neon_ActionCam_both_lk.avi')  
    
    # view_video(action_video_path=action_vid_path,
    #     action_worldtimestamps_path=action_timestamps,
    #     action_gaze_paths_dict={'LOFTR':action_gaze_path_loftr, 'LG+DISK':action_gaze_path_lg},
    #     neon_video_path=neon_vid_path,
    #     neon_worldtimestamps_path=neon_timestamps,
    #     neon_gaze_path=neon_gaze_path)  
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pupil_labs.video as plv



class VideoHandler():
    """Class to wrap video files and extract useful information from them.

    Args:
        video_dir (str): Path to the video file.
    """
    
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self._timestamps = self.get_timestamps()
        
    
    @property
    def height(self):
        with plv.open(self.video_dir) as container:
            video = container.streams.video[0]
            return video.height

    @property
    def width(self):
        with plv.open(self.video_dir) as container:
            video = container.streams.video[0]
            return video.width
    
    @property
    def timestamps(self):
        return self._timestamps
    
    def get_timestamps(self):
        with plv.open(self.video_dir) as container:
            video = container.streams[0]
            if video.type != "video":
                raise ValueError("No video stream found")
            video_timestamps = np.asarray(video.pts)
            video_timestamps = video_timestamps/video.time_base.denominator
        return np.asarray(video_timestamps, dtype=np.float32)
    
    def get_frame_by_timestamp(self, timestamp):
        timestamp = self.get_closest_timestamp(timestamp)
        timestamp_index = int(np.where(self.timestamps == timestamp)[0][0])
        with plv.open(self.video_dir) as container:
            video = container.streams[0]
            video.logger.setLevel(logging.ERROR)
            frame = video.frames[timestamp_index]
            frame = frame.to_image()
        return np.asarray(frame) 
        
    def get_timestamps_in_interval(self, start_time=0, end_time=np.inf):
        """Get all the timestamps between start_time and end_time. If no arguments are given, it returns all the timestamps.

        Args:
            start_time (float): Starting time in seconds, must be smaller than end_time. Does not necessarily correspond to the video timestamps. Defaults to 0.
            end_time (float): Ending time in seconds, must be larger than start_time. Does not necessarily correspond to the video timestamps. Defaults to np.inf.

        Returns:
            ndarray: Numpy array with all the video timestamps contained between start_time and end_time.
        """
        assert (start_time < end_time), f"Start time ({start_time} s) must be smaller than end time ({end_time} s)"
        return self.timestamps[(self.timestamps >= start_time) & (self.timestamps <= end_time)]
    
    def get_closest_timestamp(self, time):
        """Get the closest video timestamp to the given time.
        Args:
            time (float): Time in seconds

        Returns:
            float: Closest video timestamp to the given time, it can be before or after the given time.
        """
        return self.timestamps[np.argmin(np.abs(self.timestamps - time))]



def write_worldtimestamp_csv(world_timestamps, relative_timestamps, time_delay):
    """Function that creates a world timestamp csv file for action camera recording.

    Args:
        world_timestamps (str): Path to the world_timestamps.csv of the Neon recording
        relative_timestamps (ndarray): Timestamps of the action camera recording, obtainened from the metadata of the video file.
        time_delay (float): Time delay between the action camera and the Neon Scene camera in seconds. 
    """
    world_timestamps = pd.read_csv(world_timestamps)
    action_timestamps = (relative_timestamps + time_delay)/1e-9
    action_timestamps = np.int64(action_timestamps)
    action_timestamps += world_timestamps["timestamp [ns]"].iloc[0]
    action_world_timestamps = pd.DataFrame({"section id":[None for _ in action_timestamps],"record id": [None for _ in action_timestamps],"timestamp [ns]": action_timestamps})

    last_ts=max(world_timestamps['timestamp [ns]'])
    first_ts=min(world_timestamps['timestamp [ns]'])
    for section in world_timestamps['section id'].unique():
        start_section = min(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        end_section = max(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        action_world_timestamps.loc[(action_world_timestamps['timestamp [ns]']>=start_section)&(action_world_timestamps['timestamp [ns]']<end_section), 'section id'] = section
    action_world_timestamps.loc[(action_world_timestamps['section id'].isnull()) & (action_world_timestamps['timestamp [ns]']<first_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]' == first_ts]['section id']].values[0]
    action_world_timestamps.loc[(action_world_timestamps['section id'].isnull()) & (action_world_timestamps['timestamp [ns]']>=last_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]' == last_ts]['section id']].values[0]

    for record in world_timestamps['record id'].unique():
        start_record = min(world_timestamps[world_timestamps['record id'] == record]['timestamp [ns]'])
        end_record = max(world_timestamps[world_timestamps['record id'] == record]['timestamp [ns]'])
        action_world_timestamps.loc[(action_world_timestamps['timestamp [ns]']>=start_record)&(action_world_timestamps['timestamp [ns]']<end_record), 'record id'] = record
    action_world_timestamps.loc[(action_world_timestamps['record id'].isnull()) & (action_world_timestamps['timestamp [ns]']<first_ts), 'record id'] = world_timestamps.loc[world_timestamps['timestamp [ns]' == first_ts]['record id']].values[0]
    action_world_timestamps.loc[(action_world_timestamps['record id'].isnull()) & (action_world_timestamps['timestamp [ns]']>=last_ts), 'record id'] = world_timestamps.loc[world_timestamps['timestamp [ns]' == last_ts]['record id']].values[0]
    action_world_timestamps.to_csv("action_camera_world_timestamps.csv", index=False)
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
        self.video_container = plv.open(video_dir)
        self._timestamps = self.get_timestamps()
        
    
    @property
    def height(self):
        return self.video_container.streams.video[0].height
        

    @property
    def width(self):
        return self.video_container.streams.video[0].width
    
    @property
    def timestamps(self):
        return self._timestamps
    
    def get_timestamps(self):
        video = self.video_container.streams[0]
        if video.type != "video":
            raise ValueError("No video stream found")
        video_timestamps = np.asarray(video.pts)
        video_timestamps = video_timestamps/video.time_base.denominator
        return np.asarray(video_timestamps, dtype=np.float32)
    
    def get_frame_by_timestamp(self, timestamp):
        timestamp = self.get_closest_timestamp(timestamp)
        timestamp_index = int(np.where(self.timestamps == timestamp)[0][0])
        video = self.video_container.streams[0]
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
    
    def get_timestamps_around_time(self,time):
        """Get the video timestamps before and after the given time.

        Args:
            time (float): Time in seconds

        Returns:
            tuple: Tuple with the timestamps before and after the given time.
        """
        closest_timestamp = self.get_closest_timestamp(time)
        #si la closest_timestamp es mayor que el tiempo dado, entonces el timestamp anterior es el anterior al closest_timestamp
        if closest_timestamp > time:
            previous_timestamp = self.timestamps[np.where(self.timestamps == closest_timestamp)[0][0]-1]
            next_timestamp = closest_timestamp
        else:
            previous_timestamp = closest_timestamp
            next_timestamp = self.timestamps[np.where(self.timestamps == closest_timestamp)[0][0]+1]
        return previous_timestamp, next_timestamp



def write_worldtimestamp_csv(world_timestamps_dir, relative_timestamps, time_delay):
    """Function that creates a world timestamp csv file for action camera recording. The csv file is saved in the same directory as the world_timestamps.csv of the Neon recording.

    Args:
        world_timestamps_dir (str): Path to the world_timestamps.csv of the Neon recording
        relative_timestamps (ndarray): Timestamps of the action camera recording, obtained from the metadata of the video file.
        time_delay (float): Time delay between the action camera and the Neon Scene camera in seconds. 
    """
    world_timestamps = pd.read_csv(world_timestamps_dir)
    columns_for_mapping = world_timestamps.columns
    action_timestamps = (relative_timestamps + time_delay)/1e-9
    action_timestamps = np.int64(action_timestamps)
    action_timestamps += world_timestamps["timestamp [ns]"].iloc[0]
    action_world_timestamps = pd.DataFrame.from_dict({col:[None for _ in action_timestamps] for col in columns_for_mapping})
    action_world_timestamps['timestamp [ns]'] = action_timestamps
    last_ts=max(world_timestamps['timestamp [ns]'])
    first_ts=min(world_timestamps['timestamp [ns]'])

    
    for section in world_timestamps['section id'].unique():
        start_section = min(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        end_section = max(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        action_world_timestamps.loc[(action_world_timestamps['timestamp [ns]']>=start_section)&(action_world_timestamps['timestamp [ns]']<end_section), 'section id'] = section
    action_world_timestamps.loc[(action_world_timestamps['section id'].isnull()) & (action_world_timestamps['timestamp [ns]']<first_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]'] == first_ts,'section id'].values[0]
    action_world_timestamps.loc[(action_world_timestamps['section id'].isnull()) & (action_world_timestamps['timestamp [ns]']>=last_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]'] == last_ts,'section id'].values[0]

    for recording in world_timestamps['recording id'].unique():
        start_recording = min(world_timestamps[world_timestamps['recording id'] == recording]['timestamp [ns]'])
        end_recording = max(world_timestamps[world_timestamps['recording id'] == recording]['timestamp [ns]'])
        action_world_timestamps.loc[(action_world_timestamps['timestamp [ns]']>=start_recording)&(action_world_timestamps['timestamp [ns]']<end_recording), 'recording id'] = recording
    action_world_timestamps.loc[(action_world_timestamps['recording id'].isnull()) & (action_world_timestamps['timestamp [ns]']<first_ts), 'recording id'] = world_timestamps.loc[world_timestamps['timestamp [ns]'] == first_ts,'recording id'].values[0]
    action_world_timestamps.loc[(action_world_timestamps['recording id'].isnull()) & (action_world_timestamps['timestamp [ns]']>=last_ts), 'recording id'] = world_timestamps.loc[world_timestamps['timestamp [ns]']== last_ts,'recording id'].values[0]
    action_world_timestamps.to_csv(f"{str(Path(world_timestamps_dir).parent)}/action_camera_world_timestamps.csv", index=False)
    print(f"World timestamps for action camera recording saved at {Path(world_timestamps_dir).parent}/action_camera_world_timestamps.csv")
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pupil_labs.video as plv



class VideoHandler():
    def __init__(self, video_path):
        self.path = video_path
        self.video_stream = self.open_video()
        self._timestamps = self.get_timestamps()
        
    @property
    def height(self):
        return self.video_stream.height
        
    @property
    def width(self):
        return self.video_stream.width
    
    @property
    def timestamps(self):
        return self._timestamps
    
    @property
    def fps(self):
        return self.video_stream.average_rate.numerator/self.video_stream.average_rate.denominator
    
    def open_video(self):
        container = plv.open(self.path)
        video_stream = container.streams.video[0]
        if video_stream.type != "video":
            raise ValueError(f"No video stream found in {self.path}")
        video_stream.logger.setLevel(logging.ERROR)
        return video_stream
    
    def close_video(self):
        self.video_stream.close()

    def get_timestamps(self):
        video_timestamps = np.asarray(self.video_stream.pts)
        video_timestamps = video_timestamps/self.video_stream.time_base.denominator
        return np.asarray(video_timestamps, dtype=np.float32)
    
    def get_frame_by_timestamp(self, timestamp):
        timestamp, timestamp_index = self.get_closest_timestamp(timestamp)
        frame = self.video_stream.frames[timestamp_index]
        frame = frame.to_image()
        return np.asarray(frame) 
        
    def get_timestamps_in_interval(self, start_time=0, end_time=np.inf):
        assert (start_time < end_time), f"Start time ({start_time} s) must be smaller than end time ({end_time} s)"
        return self.timestamps[(self.timestamps >= start_time) & (self.timestamps <= end_time)]
    
    def get_closest_timestamp(self, time):
        after_index = np.searchsorted(self.timestamps, time)
        before_index= after_index - 1
        ts_after = self.timestamps[after_index]
        ts_before = self.timestamps[before_index]
        if np.abs(ts_after - time) < np.abs(ts_before - time):
            return ts_after, int(after_index)
        else:
            return ts_before, int(before_index)
    
    def get_surrounding_timestamps(self,time):
        closest_timestamp = self.get_closest_timestamp(time)
        if closest_timestamp > time:
            previous_timestamp = self.timestamps[np.where(self.timestamps == closest_timestamp)[0][0]-1]
            next_timestamp = closest_timestamp
        else:
            previous_timestamp = closest_timestamp
            next_timestamp = self.timestamps[np.where(self.timestamps == closest_timestamp)[0][0]+1]
        return previous_timestamp, next_timestamp



def write_worldtimestamp_csv(timestamps_path, aligned_relative_timestamps):
    """Function that creates a world timestamp csv file for action camera recording. The csv file is saved in the same directory as the world_timestamps.csv of the Neon recording.

    Args:
        timestamps_dir (str): Path to the world_timestamps.csv of the Neon recording
        aligned_relative_timestamps (ndarray): Timestamps of the action camera recording, obtained from the metadata of the video file. This function assumes that the timestamps are already aligned with the Neon recording timestamps.  
    """
    world_timestamps = pd.read_csv(timestamps_path)
    columns_for_mapping = world_timestamps.columns

    action_timestamps = aligned_relative_timestamps/1e-9
    action_timestamps = np.int64(action_timestamps)
    action_timestamps += world_timestamps["timestamp [ns]"].iloc[0]

    action_timestamps_df = pd.DataFrame.from_dict({col:[None for _ in action_timestamps] for col in columns_for_mapping})
    action_timestamps_df['timestamp [ns]'] = action_timestamps
    action_timestamps_df['recording id'] = world_timestamps['recording id'].values[0]
    last_ts=max(world_timestamps['timestamp [ns]'])
    first_ts=min(world_timestamps['timestamp [ns]'])

    
    for section in world_timestamps['section id'].unique():
        start_section = min(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        end_section = max(world_timestamps[world_timestamps['section id'] == section]['timestamp [ns]'])
        action_timestamps_df.loc[(action_timestamps_df['timestamp [ns]']>=start_section)&(action_timestamps_df['timestamp [ns]']<end_section), 'section id'] = section

    action_timestamps_df.loc[(action_timestamps_df['section id'].isnull()) & (action_timestamps_df['timestamp [ns]']<first_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]'] == first_ts,'section id'].values[0]
    action_timestamps_df.loc[(action_timestamps_df['section id'].isnull()) & (action_timestamps_df['timestamp [ns]']>=last_ts), 'section id'] = world_timestamps.loc[world_timestamps['timestamp [ns]'] == last_ts,'section id'].values[0]

    saving_path = Path(timestamps_path).parent
    action_timestamps_df.to_csv(Path(saving_path,'action_camera_world_timestamps.csv'), index=False)
    print(f"Timestamps for action camera recording saved at {Path(saving_path/'action_camera_world_timestamps.csv')}")
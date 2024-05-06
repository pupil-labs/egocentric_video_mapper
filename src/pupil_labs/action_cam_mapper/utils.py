import av
import imageio.v3 as iio
import numpy as np
import pandas as pd



class VideoHandler():
    
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.timestamps=self.get_timestamps()
        self.height, self.width = self.get_video_dimensions()
    
    def get_timestamps(self):
        with av.open(self.video_dir) as container:
            video = container.streams.video[0]
            video_timestamps = [
                packet.pts / video.time_base.denominator for packet in container.demux(video) if packet.pts is not None
            ]
        return np.asarray(video_timestamps)

    def get_video_dimensions(self):
        return iio.improps(self.video_dir, plugin="pyav").shape[1], iio.improps(self.video_dir, plugin="pyav").shape[2]
    
    def get_frame_by_timestamp(self, timestamp):
        timestamp = self.get_closest_timestamp(timestamp)
        timestamp_index = np.where(self.timestamps == timestamp)[0][0]
        frame = iio.imread(
            self.video_dir,
                index=timestamp_index,
                plugin="pyav",
            )
        return frame
        
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
    pass
    # creates world timestamps from relative timestamps and a time delay
    # world_timestamps is the csv file from neon scene camera
    # relative_timestamps is the timestamps of the external camera obtained from the .mp4 metadata
    # time_delay is the time delay between the two cameras, calculated using alignSignals
    # new_world_timestamps = (relative_timestamps + time_delay)/10e-9 + world_timestamps[timestamp [ns]][0]
    # after getting the new world timestamps, the other columns in the csv file are parsed to the new timestamps
    # the new csv file is saved in the same directory as the original csv file under the name external_camera_world_timestamps.csv

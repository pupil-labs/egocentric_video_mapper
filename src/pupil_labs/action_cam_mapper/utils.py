import numpy as np
import pandas as pd
import av


# functions there are common to all modules or that are helpers

class VideoHandler():
    #class that handles video reading, frame retrieval and relative timestamps
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.timestamps=self.get_timestamps()
        self.width, self.height = self.get_video_dimensions()
    
    def get_timestamps(self):
        with av.open(self.video_dir) as container:
            video = container.streams.video[0]
            video_timestamps = [
                packet.pts / video.time_base.denominator for packet in container.demux(video) if packet.pts is not None
            ]
        self.timestamps = video_timestamps

    def get_frame_by_timestamp(self, timestamp):
        pass
        # according to PyAV obscure documentation, the seek function is the way to go
        # seek the frame at the timestamp in stream.time_base units
        # https://github.com/PyAV-Org/PyAV/discussions/1113
    
    def get_timestamps_in_interval(self, start_time, end_time):
        # get all the timestamps between start_time and end_time
        # start_time and end_times are in seconds and do not necessarily correspond to the video timestamps
        assert (start_time < end_time), f"Start time ({start_time} s) must be smaller than end time ({end_time} s)"
        return self.timestamps[start_time<=self.timestamps<=end_time]
    
    def get_closest_timestamp(self, time):
        # get the closest video timestamp to the given timestamp 
        return self.timestamps[np.argmin(np.abs(np.array(self.timestamps) - time))]



def write_worldtimestamp_csv(world_timestamps, relative_timestamps, time_delay):
    pass
    # creates world timestamps from relative timestamps and a time delay
    # world_timestamps is the csv file from neon scene camera
    # relative_timestamps is the timestamps of the external camera obtained from the .mp4 metadata
    # time_delay is the time delay between the two cameras, calculated using alignSignals
    # new_world_timestamps = (relative_timestamps + time_delay)/10e-9 + world_timestamps[timestamp [ns]][0]
    # after getting the new world timestamps, the other columns in the csv file are parsed to the new timestamps
    # the new csv file is saved in the same directory as the original csv file under the name external_camera_world_timestamps.csv

# would it be of interest to create a module that calls the functions and optic flow objects so it is all done in one go?


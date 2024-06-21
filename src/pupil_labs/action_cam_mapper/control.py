import linecache
import os
import time
import logging
import tracemalloc
# from utils import VideoHandler
import numpy as np
import pupil_labs.video as plv
# from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
            % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main():
    tracemalloc.start()

    
    action_vid_path='/users/sof/gaze_mapping/raw_videos/InstaVid/wearingNeon_2m/AVun_20240216_160246_055.mp4'
    neon_vid_path='/users/sof/gaze_mapping/raw_videos/Neon/Raw_Data/2024-02-16_wearingNeon/2024-02-16_15-58-13-6310bec3/cc00c32d_0.0-146.071.mp4'
    neon_timestamps='/users/sof/gaze_mapping/raw_videos/Neon/Raw_Data/2024-02-16_wearingNeon/2024-02-16_15-58-13-6310bec3/world_timestamps.csv'

    
    with plv.open(action_vid_path) as container:
        container.streams[0].logger.setLevel(logging.ERROR) # not this
        for timestamp_index in range(0, 150):
            time.sleep(0.3)
            frame = container.streams[0].frames[timestamp_index]
        
    # # run video handler code
    # actionVid = VideoHandler(video_dir=action_vid_path)
    # # neonVid=VideoHandler(video_dir=neon_vid_path)
    # # neon_H=neonVid.height
    # # neon_W=neonVid.width
    # action_H=actionVid.height
    # action_W=actionVid.width
    # # neon_timestamps=neonVid.timestamps
    # action_timestamps=actionVid.timestamps
    # # neonframe = neonVid.get_frame_by_timestamp(neonVid.timestamps[44])
    # # actionframe = actionVid.get_frame_by_timestamp(actionVid.timestamps[44])
    # selected_timestamps=actionVid.get_timestamps_in_interval(0,5)
    # for i in selected_timestamps:
    #     action_frame = actionVid.get_frame_by_timestamp(i)
    # action_of = OpticFlowCalculatorLK(video_dir=action_vid_path,grid_spacing=100)
    # # neon_of = OpticFlowCalculatorLK(video_dir=neon_vid_path,grid_spacing=100)
    # ac_result = action_of.get_optic_flow(0,10)
    # # neon_result = neon_of.get_optic_flow(0,5)
    print('time to sleep')
    time.sleep(30)
    print('woke up')
    
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot,limit=5)

if __name__ == "__main__":
    main()
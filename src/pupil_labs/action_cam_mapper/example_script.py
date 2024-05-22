import os
import pandas as pd
from optic_flow import OpticFlowCalculatorLK, OpticFlowCalculatorFarneback
from utils import VideoHandler, write_worldtimestamp_csv
from sync_videos import OffsetCalculator
from gaze_mapper import ActionCameraGazeMapper
from pathlib import Path



def get_file(folder_path, file_suffix='.mp4', required_in_name='0'):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(folder_path)
            for name in files
            if name.endswith(file_suffix) and required_in_name in name][0]

def calc_optic(neon_video,action_video,name,of_choice='farneback'):
    Path(f'/users/sof/optic_flow/{name}').mkdir(parents=True, exist_ok=True)
    if of_choice.lower() == 'farneback':
        action_of = OpticFlowCalculatorFarneback(video_dir=action_video)
        optic_flow_action=action_of.get_all_optic_flow()
        neon_of = OpticFlowCalculatorFarneback(video_dir=neon_video)
        optic_flow_neon = neon_of.get_all_optic_flow()
    elif of_choice.lower() == 'lk':
        action_of = OpticFlowCalculatorLK(video_dir=action_video)
        optic_flow_action=action_of.get_all_optic_flow()
        neon_of = OpticFlowCalculatorLK(video_dir=neon_video)
        optic_flow_neon = neon_of.get_all_optic_flow()
    else:
        raise ValueError('Invalid optic flow choice. Choose from "farneback" or "lk"')
    action_of.write_to_csv(output_file=f'/users/sof/optic_flow/{name}/{name}_action_{of_choice}_of.csv')
    neon_of.write_to_csv(output_file=f'/users/sof/optic_flow/{name}/{name}_neon_{of_choice}_of.csv')

def align_videos(action_result,neon_result,action_vid_path,neon_timestamps):
    offset_calc=OffsetCalculator(source=action_result['angle'].values,source_timestamps=action_result['start'].values, destination=neon_result['angle'].values, destination_timestamps=neon_result['start'].values,resampling_frequency=500)
    t_offset, pearson_corr = offset_calc.estimate_time_offset()
    print(f'Estimated offset: {t_offset} seconds (Pearson correlation: {pearson_corr})')
    actionVid=VideoHandler(action_vid_path)
    write_worldtimestamp_csv(world_timestamps_dir=neon_timestamps, relative_timestamps=actionVid.timestamps, time_delay=t_offset)

def main_mapper(action_vid_path,
                neon_vid_path,
                neon_timestamps,
                action_timestamps,
                neon_gaze_csv,
                name,
                scene_location='indoor',
                ):
    param={'location':scene_location, 'gpu_num':0}
    mapper = ActionCameraGazeMapper(neon_gaze_csv=neon_gaze_csv,
        neon_video_dir=neon_vid_path,
        action_video_dir=action_vid_path,
        neon_worldtimestamps=neon_timestamps,
        action_worldtimestamps=action_timestamps,
        image_matcher='loftr',
        image_matcher_parameters=param,
        neon_opticflow_csv=f'/users/sof/optic_flow/{name}/{name}_neon_lk_of.csv',
        action_opticflow_csv=f'/users/sof/optic_flow/{name}/{name}_action_lk_of.csv',
        patch_size=1000)
    Path(f'/users/sof/mapped_gaze/{name}').mkdir(parents=True, exist_ok=True)
    mapper.map_gaze(saving_path=f'/users/sof/mapped_gaze/{name}/{name}_gaze_mapping.csv')

def main(action_vid_path, neon_timeseries_dir, optic_flow_choice='lk', location='indoor'):
    neon_vid_path=get_file(neon_timeseries_dir, file_suffix='.mp4')
    neon_timestamps=neon_timeseries_dir+'/world_timestamps.csv'
    neon_gaze_csv=neon_timeseries_dir+'/gaze.csv'
    name=Path(action_vid_path).parent.name
    #Step 1: Calculate optic flow
    calc_optic(neon_video=neon_vid_path,action_video=action_vid_path,name=name,of_choice=optic_flow_choice)
    print('Optic flow for both videos calculated')
    #Step 2: Estimate time offset and create action world timestamps
    action_result=pd.read_csv(f'/users/sof/optic_flow/{name}/{name}_action_{optic_flow_choice}_of.csv')
    neon_result=pd.read_csv(f'/users/sof/optic_flow/{name}/{name}_neon_{optic_flow_choice}_of.csv')
    align_videos(action_result,neon_result,action_vid_path,neon_timestamps)
    #check is world_timestamps.csv is created
    action_timestamps=neon_timeseries_dir+'/action_camera_world_timestamps.csv'
    if not Path(action_timestamps).exists():
        raise FileNotFoundError(f'{action_timestamps} not created!')
    #Step 3: Map gaze
    main_mapper(action_vid_path=action_vid_path,
                neon_vid_path=neon_vid_path,
                neon_timestamps=neon_timestamps,
                action_timestamps=action_timestamps,
                neon_gaze_csv=neon_gaze_csv,
                name=name,
                scene_location=location,
                )

if __name__ == "__main__":
    action_vid_path='/users/sof/gaze_mapping/raw_videos/InstaVid/wearingNeon_2m/AVun_20240216_160246_055.mp4'
    neon_timeseries_path='/users/sof/gaze_mapping/raw_videos/Neon/Raw_Data/2024-02-16_wearingNeon/2024-02-16_15-58-13-6310bec3/'
    main(action_vid_path=action_vid_path, neon_timeseries_dir=neon_timeseries_path, optic_flow_choice='lk', location='indoor')
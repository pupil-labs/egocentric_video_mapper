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

def calc_optic(neon_video,action_video,output_dir,of_choice='farneback'):
    output_dir = Path(output_dir,'optic_flow')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    
    action_saving_path = Path(output_dir,f'action_{of_choice}_of.csv')
    neon_saving_path = Path(output_dir,f'neon_{of_choice}_of.csv')
    action_of.write_to_csv(output_file=action_saving_path)
    neon_of.write_to_csv(output_file=neon_saving_path)
    return action_saving_path,neon_saving_path

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
                neon_opticflow_csv,
                action_opticflow_csv,
                output_dir,
                matcher,
                ):
    output_dir=Path(output_dir,f'mapped_gaze/{matcher["choice"]}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    param=matcher['parameters']
    print(f'Using {matcher["choice"]} with parameters: {param}')

    mapper = ActionCameraGazeMapper(neon_gaze_csv=neon_gaze_csv,
        neon_video_dir=neon_vid_path,
        action_video_dir=action_vid_path,
        neon_worldtimestamps=neon_timestamps,
        action_worldtimestamps=action_timestamps,
        image_matcher=matcher['choice'],
        image_matcher_parameters=param,
        neon_opticflow_csv=neon_opticflow_csv,
        action_opticflow_csv=action_opticflow_csv,
        patch_size=1000)
    
    mapper.map_gaze(saving_path=Path(output_dir,'action_gaze.csv'))

def main(action_vid_path, neon_timeseries_dir, output_dir, image_matcher,optic_flow_choice='lk',render_video=False):

    neon_vid_path=get_file(neon_timeseries_dir, file_suffix='.mp4')
    neon_timestamps=neon_timeseries_dir+'/world_timestamps.csv'
    neon_gaze_csv=neon_timeseries_dir+'/gaze.csv'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #Step 1: Calculate optic flow
    action_of_path, neon_of_path=calc_optic(neon_video=neon_vid_path,action_video=action_vid_path,output_dir=output_dir,of_choice=optic_flow_choice)
    print('Optic flow for both videos calculated')

    #Step 2: Estimate time offset and create action world timestamps
    action_result=pd.read_csv(action_of_path)
    neon_result=pd.read_csv(neon_of_path)
    align_videos(action_result,neon_result,action_vid_path,neon_timestamps)

    #check if world_timestamps.csv is created
    action_timestamps=neon_timeseries_dir+'/action_camera_world_timestamps.csv'
    if not Path(action_timestamps).exists():
        raise FileNotFoundError(f'{action_timestamps} not created!')
    
    #Step 3: Map gaze
    main_mapper(action_vid_path=action_vid_path,
                neon_vid_path=neon_vid_path,
                neon_timestamps=neon_timestamps,
                action_timestamps=action_timestamps,
                neon_gaze_csv=neon_gaze_csv,
                neon_opticflow_csv=neon_of_path,
                action_opticflow_csv=action_of_path,
                output_dir=output_dir,
                matcher=image_matcher
                )
    #Step 4 (Optional): Render simultaneous videos with gaze in both
    if render_video:
        pass


if __name__ == "__main__":
    action_vid_path='/users/sof/gaze_mapping/raw_videos/InstaVid/wearingNeon_2m/AVun_20240216_160246_055.mp4'
    neon_timeseries_path='/users/sof/gaze_mapping/raw_videos/Neon/Raw_Data/2024-02-16_wearingNeon/2024-02-16_15-58-13-6310bec3/'

    output_dir='/users/sof/action_map_experiments/' # parent directory for all outputs
    name=Path(action_vid_path).parent.name
    output_dir=Path(output_dir,name)

    param_lg = {'num_features':2048,'gpu_num':0}
    param_loftr = {'location':'indoor', 'gpu_num':0}
    image_matcher_loftr={'choice':'loftr','parameters':param_loftr}
    image_matcher_lg={'choice':'disk_lightglue','parameters':param_lg}

    main(action_vid_path=action_vid_path, 
        neon_timeseries_dir=neon_timeseries_path,
        output_dir=output_dir, 
        image_matcher=image_matcher_lg,
        optic_flow_choice='lk', 
        render_video=False)
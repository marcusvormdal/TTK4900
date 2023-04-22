import numpy as np
import cv2
import torch
from datetime import datetime
from time import process_time

from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.track import Track
from stonesoup.plotter import Plotterly
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

import lidar_driver.lidar_driver as ld
import camera_driver.camera_driver as cd
import jpda_driver.jpda_driver as jd
import support_functions.support_functions as sf
import plot_driver.plot_driver as pd

def run(start_stamp):
    # Control variables
    use_capture = True
    #1675168055 #- corner  1675168005-wall?  #  -
    
    #3615 for trash bag
    video_path = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/full_run.mp4'
    # Initiate plotting
    # Initiate tracks
    start_time = datetime.now()
    prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    
    #global variables
    lidar_measurements = []
    current_lines = np.array([[3,[0.0,0.0,0.0,0.0]]], dtype=object)
    thresholded_raw = []
    detections = []
    track = []
    last_position = [0,0,0]
    position_delta = [0,0,0]
    lm_plot = []
    ned_track = []
    # Initiate LSD
    detector = cv2.createLineSegmentDetector(0)

    # If using captured data
    if use_capture == True:
        #ld.read_pcap_data('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-13-18-08_Velodyne-VLP-16-Data.pcap')   # Redo lidar_data
        
        # load data
        raw_lidar_data = np.load('./lidar_driver/lidar_trash_point_array.npy', allow_pickle=True)
        video = cv2.VideoCapture(video_path)
        model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov5', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/pLitterFloat_800x752_to_640x640.pt', source='local', force_reload=True)
        pos_stream = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/testrecord.txt'
        gps_date = sf.get_gps_date_ts(pos_stream)  # Need to resolve once

        # cast generators
        lidar_generator = ld.get_raw_lidar_data(raw_lidar_data, start_stamp)
        camera_generator = cd.get_camera_frame(video, start_stamp, video_path) 
        pos_generator = sf.get_position(pos_stream, gps_date, start_stamp)

        # Initialize data frames
        curr_lidar = next(lidar_generator)
        curr_cam = next(camera_generator)
        curr_pos = next(pos_generator)
        
        #Track initialization 
        last_position = curr_pos[1]
        
        measurement_model = LinearGaussian(ndim_state=8, mapping=[0, 2, 4, 6],
                                   noise_covar=np.diag([1**2, 1**2, 1**2, 1**2]))
    while(True):
        t1_start = process_time() 
        data_type, ts, data, curr_lidar, curr_cam, curr_pos = sf.data_handler(curr_lidar, curr_cam, curr_pos, lidar_generator, camera_generator, pos_generator)
        
        if data_type == 'lid':
            lidar_measurements, current_lines, thresholded_raw = ld.get_lidar_measurements(detector, data, position_delta = position_delta, radius = 10, intensity=0, heigth=-0.80, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
           
            current_lines[:,1] = sf.get_relative_pos(current_lines[:,1], 'line')
            current_lines[:,1] = ld.update_lines_pos(7, current_lines[:,1])
            lidar_measurements = ld.cluster_measurements(lidar_measurements)
            lm_plot = ld.set_lidar_offset(5, np.copy(lidar_measurements))
            data = ld.set_lidar_offset(last_position[2], np.copy(lidar_measurements), t = [0.0+last_position[0],0.0+last_position[1]])
            try:
                for meas in data[:,0:2]:
                    ned_track.append([meas,'lid'])
            except:
                pass  
        elif data_type == 'cam':
            detections = cd.detect_trash(data, model, [5,0,-10]) #last_position[2]
            data = cd.set_cam_offset(last_position[2], detections,[0.10+last_position[0],0.0+last_position[1]])
            #print(detections)
            #print(last_position[2])
            #print(data)
            if data != []:
                try:
                    for meas in data:
                        ned_track.append([meas,'cam'])
                except:
                    pass
        elif data_type == 'pos':
            position_delta = np.array(data) - np.array(last_position)
            last_position = data
            track.append(last_position)
            curr_track = np.copy(track)
            
        t1_stop = process_time()
        
        print(data_type, ' : ', t1_stop-t1_start, ' : ', ts)
        
        state_vector = StateVector([0,0,0,0,0,0,0,0])
        tracker_data = Detection(state_vector =state_vector, timestamp =ts, measurement_model = measurement_model)
        
        plot_data = [data_type, thresholded_raw, lm_plot,current_lines, detections, curr_track, curr_cam[1], np.copy(ned_track)]
        
        yield ts, tracker_data, data_type, plot_data
        
def main():
    start_stamp = 1675168347
    ts = start_stamp
    runner = run(start_stamp)
    runtime = start_stamp + int(input("Runtime (s): "))
    animation_data = []
    
    while runtime > ts:
        ts, data, data_type, plot_data = next(runner)
        animation_data.append(plot_data)
        
    #tracker = jd.track(runner)
    #for timestamp, tracks in tracker:
    #    print(timestamp)
    #    print(tracks)
    #while True:
        #data_type = next(runner)
        
    pd.animate(animation_data)     
    
main()  
    

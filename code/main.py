import numpy as np
import cv2
import torch
import matplotlib as lib
import matplotlib.pyplot as plt
from datetime import datetime

from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


import lidar_driver.lidar_driver as ld
import camera_driver.camera_driver as cd
import jpda_driver.jpda_driver as jd
import support_functions.support_functions as sf
import plot_driver.plot_driver as pd

def main():
    # Control variables
    use_capture = True
    start_frame = 3615 
    #3615 for trash bag
    
    # Initiate plotting
    fig, ax = plt.subplots(2,2)
    # Initiate tracks
    start_time = datetime.now()
    prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    tracks = {Track([prior1]), Track([prior2])}
    #global variables
    lidar_measurements = []
    current_lines = []
    thresholded_raw = []
    draw_lines = []
    camera_bounds = []
    predictions = []
    start_pos = None
    position = None
    last_position = None
    position_delta = None

    # Initiate LSD
    detector = cv2.createLineSegmentDetector(0)

    # If using captured data
    if use_capture == True:
        #ld.read_pcap_data('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-12-48-25_Velodyne-VLP-16-Data.pcap')   # Redo lidar_data
        
        # load data
        raw_lidar_data = np.load('./lidar_driver/lidar_trash_point_array.npy', allow_pickle=True)
        video = cv2.VideoCapture('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/trash_collect.mp4')
        model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov5', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/pLitterFloat_800x752_to_640x640.pt', source='local', force_reload=True)
        pos_stream = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/testrecord.txt'
        gps_date = sf.get_gps_date_ts(pos_stream)  # Need to reseolve once

        # cast generators
        lidar_generator = ld.get_raw_lidar_data(raw_lidar_data, start_frame = start_frame)
        camera_generator = cd.get_camera_frame(video, start_frame = start_frame) 
        pos_generator = sf.get_position(pos_stream, gps_date,  start_frame = start_frame, start_pos = start_pos)
        
        # Initialize data frames
        curr_lidar = next(lidar_generator)
        curr_cam = next(camera_generator)
        curr_pos = next(pos_generator)
        
        #Track initialization 
        pos_track = [curr_pos[1]]
        last_position = curr_pos[1]
           
    while(True):
        
        data_type, data, curr_lidar, curr_cam, curr_pos = sf.data_handler(curr_lidar, curr_cam, curr_pos, lidar_generator, camera_generator, pos_generator)
        
        if data_type == 'lidar':
            lidar_measurements, current_lines, thresholded_raw, draw_lines  = ld.get_lidar_measurements(detector, data, position_delta = position_delta, radius = 10, intensity=0, heigth=-0.65, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
     
        elif data_type == 'cam':
            predictions, camera_bounds = cd.detect_trash(data, model)

        elif data_type == 'pos':
            print("data", data)
            pos_track.append(data)
            position_delta = np.array(data) - np.array(last_position)
            last_position = data
         
        #tracks = jd.JPDA(start_time, tracks, measurements)
        pd.full_plotter(ax, detector, thresholded_raw, lidar_measurements, current_lines, draw_lines, camera_bounds, predictions, pos_track, camera_frame=curr_cam[1]) 
    
    
    '''
    for i in range(np.shape(raw_lidar_data)[0]):
        print('Frame: ', i)

        lidar_frame  = next(lidar_generator)    # All lidar data points at current timestep
        camera_frame =  next(image_generator)          # Image at current timestep
        position = next(pos_generator)                # Current position (x,y,z)
        pos_track.append(position)
        predictions, camera_bounds = cd.detect_trash(camera_frame[1], model)
        position_delta = np.array(position) - np.array(last_position)
        last_position = position
        lidar_measurements, current_lines, thresholded_raw, draw_lines  = ld.get_lidar_measurements(detector, lidar_frame[1], position_delta = position_delta, radius = 10, intensity=0, heigth=-0.65, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
        
        #measurements = np.concatenate([lidar_measurements, camera_measurements], axis = 0)  # Treat equally? Might want to weigh differently
        
        #tracks = jd.JPDA(start_time, tracks, measurements)
        pd.full_plotter(ax, detector, thresholded_raw, lidar_measurements, current_lines, draw_lines,  camera_bounds, predictions, pos_track, camera_frame=camera_frame[1]) 
    '''
main()
    

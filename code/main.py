import numpy as np
import cv2
import torch
import matplotlib as lib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from time import process_time

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
    start_stamp = 1675167671
    #3615 for trash bag
    video_path = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/full_run.mp4'
    # Initiate plotting
    
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, :])
    ax =[ax1, ax2, ax3]
    
    pd.initiate_plot(ax)
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
    position = None
    last_position = [0,0,0]
    position_delta = [0,0,0]

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
        gps_date = sf.get_gps_date_ts(pos_stream)  # Need to reseolve once

        # cast generators
        lidar_generator = ld.get_raw_lidar_data(raw_lidar_data, start_stamp)
        camera_generator = cd.get_camera_frame(video, start_stamp, video_path) 
        pos_generator = sf.get_position(pos_stream, gps_date, start_stamp)
        
        # Initialize data frames
        curr_lidar = next(lidar_generator)
        curr_cam = next(camera_generator)
        curr_pos = next(pos_generator)
        print("Synchronizing frames")
        #Track initialization 
        last_position = curr_pos[1]
           
    while(True):
        t1_start = process_time() 
        data_type, data, curr_lidar, curr_cam, curr_pos = sf.data_handler(curr_lidar, curr_cam, curr_pos, lidar_generator, camera_generator, pos_generator)
        if data_type == 'lid':
            lidar_measurements, current_lines, thresholded_raw, draw_lines  = ld.get_lidar_measurements(detector, data, position_delta = position_delta, radius = 10, intensity=0, heigth=-0.65, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
     
        elif data_type == 'cam':
            predictions, camera_bounds = cd.detect_trash(data, model)

        elif data_type == 'pos':
            position_delta = np.array(data) - np.array(last_position)
            last_position = data
         
        #tracks = jd.JPDA(start_time, tracks, measurements)
        t1_stop = process_time()
        print(data_type, ' : ', t1_stop-t1_start)
        #t2_start = process_time() 
        #pd.full_plotter(ax, data_type, detector, thresholded_raw, lidar_measurements, current_lines, draw_lines, camera_bounds, predictions, last_position, camera_frame=curr_cam[1]) 
        #t2_stop = process_time()
        #print('Plot Time usage: ', t2_stop-t2_start)
    
main()
    

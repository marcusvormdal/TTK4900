import numpy as np
import cv2
import torch
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
    start_frame = 3200
    # Initiate plotting
    fig, ax = plt.subplots(2,2)

    # Initiate tracks
    start_time = datetime.now()
    prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    tracks = {Track([prior1]), Track([prior2])}
    
    # Initiate LSD
    detector = cv2.createLineSegmentDetector(0)
    orientation = []
    current_lines = []
    old_orientation = []
    # If using captured data
    if use_capture == True:
        # Create generator for raw lidar data
        raw_lidar_data = np.load('./lidar_driver/lidar_trash_point_array.npy', allow_pickle=True)
        lidar_frame_generator = ld.get_raw_lidar_data(raw_lidar_data, start_frame = start_frame)
        
        # Create generator for camera frame
        video = cv2.VideoCapture('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/trash_collect.mp4')
        #model = torch.hub.load('yolov5', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/pLitterFloat_800x752_to_640x640.pt', source='local', force_reload=True)
        image_generator = cd.get_camera_frame(video, start_frame = start_frame) 
 
    #while(True):
    for i in range(np.shape(raw_lidar_data)[0]):
        
        lidar_frame  =  next(lidar_frame_generator)    # All lidar data points at current timestep
        camera_frame =  next(image_generator)          # Image at current timestep
        position     =  sf.get_position()                 # Current position (x,y,z)
        old_orientation = orientation
        orientation  =  sf.get_orientation()              # Current orientation
        orientation_delta = orientation #- old_orientation
        print('Frame: ', i)
        if np.size(lidar_frame) != 0:
            lidar_measurements, current_lines, thresholded_raw  = ld.get_lidar_measurements(detector, lidar_frame, radius = 10, intensity=0, heigth=-0.25, current_lines=current_lines, orientation_delta=orientation_delta)         # All lidar points on the water surface, bounds for plotting
        
        #camera_measurements, camera_bounds = cd.get_camera_measurements(camera_frame)       # Extracted points of objects, bounds for plotting
        #measurements = np.concatenate([lidar_measurements, camera_measurements], axis = 0)  # Treat equally? Might want to weigh differently
        
        #tracks = jd.JPDA(start_time, tracks, measurements)
        
        #camera_bounds = []
        pd.full_plotter(ax, detector, thresholded_raw, lidar_measurements, current_lines, camera_bounds = [], camera_frame=camera_frame) 
    
main()
    







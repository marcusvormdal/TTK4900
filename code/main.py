import numpy as np
import cv2
import torch
from datetime import datetime
from time import process_time
from PIL import Image
from stonesoup.types.array import StateVector
from stonesoup.plotter import Plotterly
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
import plotly
from io import BytesIO
import base64

import lidar_driver.lidar_driver as ld
import camera_driver.camera_driver as cd
import jpda_driver.jpda_driver as jd
import support_functions.support_functions as sf
import plot_driver.plot_driver as pd
import copy
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # This is a band-aid solution
def run(start_stamp, runtime):
    # Control variables
    use_capture = True
    #1675168055 #- corner  1675168005-wall?  #  -
    
    #3615 for trash bag
    video_path = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/full_run.mp4'

    #global variables
    ts = start_stamp
    lidar_measurements = []
    current_lines = np.array([[3,[0.0,0.0,0.0,0.0]]], dtype=object)
    thresholded_raw = []
    detections = []
    track = []
    last_position = [0,0,0]
    position_delta = [0,0,0]
    lm_plot = []
    ned_track = []
    lidar_buffer = []
    buffer_index = 0
    
    # Initiate LSD
    
    detector = cv2.createLineSegmentDetector(0)

    # If using captured data
    if use_capture == True:
        #ld.read_pcap_data('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-13-18-08_Velodyne-VLP-16-Data.pcap')   # Redo lidar_data
        
        # load data
        raw_lidar_data = np.load('./lidar_driver/lidar_trash_point_array.npy', allow_pickle=True)
        video = cv2.VideoCapture(video_path)
        model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov7', 'custom', path_or_model = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolo7.pt', source='local')
        #model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov7', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolo7_light.pt', source='local', force_reload=False)
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
        measurement_model_cam = LinearGaussian(ndim_state=4, mapping=[0,2], noise_covar=np.diag([0.5**2, 0.5**2]))
        measurement_model_lid = LinearGaussian(ndim_state=4, mapping=[0,2], noise_covar=np.diag([0.5**2, 0.5**2]))
    while(runtime > ts):
        t1_start = process_time() 
        tracker_data = set()
        data_type, ts, data, curr_lidar, curr_cam, curr_pos = sf.data_handler(curr_lidar, curr_cam, curr_pos, lidar_generator, camera_generator, pos_generator)
        
        if data_type == 'lid':
            lidar_measurements, current_lines, thresholded_raw = ld.get_lidar_measurements(detector, data, position_delta = position_delta, radius = 10, intensity=0, heigth=-0.80, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
            if np.size(current_lines) != 0:
                current_lines[:,1] = sf.get_relative_pos(current_lines[:,1], 'line')
                current_lines[:,1] = ld.update_lines_pos(0, current_lines[:,1])
            lm_plot = lidar_measurements #ld.set_lidar_offset(0,copy.deepcopy(lidar_measurements))
            #test = np.array([[10.0,5.0],[10.0,0.0]])
            data = ld.set_lidar_offset(0+last_position[2], copy.deepcopy(lidar_measurements), t = [0.0+last_position[0],0.0+last_position[1]])
            if np.size(data) != 0:
                for meas in data:
                    lidar_buffer.append(meas)
                    
            if buffer_index == 3:  # Buffer lidar measurements and cluster after
                clustered = ld.cluster_measurements(lidar_buffer)
                for c in clustered:
                    ned_track.append([c, 'lid'])
                    tracker_data.add(Detection(state_vector =StateVector([c[1],c[0]]), timestamp =datetime.fromtimestamp(ts), measurement_model = measurement_model_cam))

                lidar_buffer = []
                buffer_index = 0

            else:
                buffer_index += 1
                t1_stop = process_time()
                print(data_type, ' : ', t1_stop-t1_start, ' : ', ts)
                continue
            
        elif data_type == 'cam':
            detections = cd.detect_trash(data, model, [1.5,0.0,-2.7]) #-2.55 best
            data = cd.set_cam_offset(last_position[2], detections,[0.0+last_position[0],0.0+last_position[1]])
            if np.size(data) != 0:
                for meas in data:
                    ned_track.append([meas,'cam'])
                    tracker_data.add(Detection(state_vector = StateVector([meas[1],meas[0]]), timestamp =datetime.fromtimestamp(ts), measurement_model = measurement_model_lid))

        elif data_type == 'pos':
            position_delta = np.array(data) - np.array(last_position)
            last_position = data
            #print("current pos", last_position[2])
            track.append(last_position)
            curr_track = np.copy(track)
            continue
        t1_stop = process_time()
        
        print(data_type, ' : ', t1_stop-t1_start, ' : ', ts)
        
        plot_data = [data_type, thresholded_raw, lm_plot,current_lines, detections, curr_track, curr_cam[1], np.copy(ned_track)]
        
        yield ts, tracker_data, data_type, plot_data

def detector_wrapper(gen):
    for ts, tracker_data, _, _ in gen:
        yield datetime.fromtimestamp(ts), tracker_data
        
def main():
    start_stamp = 1675168544 #1675168352
    runtime = start_stamp + int(input("Runtime (s): "))
    animation_data = []
    jpda = True
    tracks = set()
    if jpda == False:
        runner = run(start_stamp, runtime)
        for ts, data, data_type, plot_data in runner:
                animation_data.append(plot_data)
                
        pd.animate(animation_data)     
    else:
        runner = run(start_stamp, runtime)
        gen = detector_wrapper(runner)
        tracker = jd.track(gen)
        for _, ctracks in tracker:
            tracks.update(ctracks)
        plotter = Plotterly()

        pil_img = Image.open("brattorkaia.png")
        prefix = "data:image/png;base64,"
        with BytesIO() as stream:
            pil_img.save(stream, format="png")
            base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
        
        plotter.plot_tracks(tracks, [0, 2], uncertainty=False)
        plotter.fig.add_traces([plotly.graph_objects.Image(source=base64_string)])
         
        plotter.fig.show()

        
main()  
    

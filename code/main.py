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
import rosbag
import pymap3d
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # Removing internal opencv error

def run(start_stamp, runtime, ros = True, relative_pos = True):
    use_capture = True    
    video_path = '../data/lidar_collection_31_01/videos/full_run.mp4'
    bag = rosbag.Bag('./bags/pbox_1.bag', "r")
    
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
    curr_track = [[0,0],[0,0]]
    # Initiate LSD
    detector = cv2.createLineSegmentDetector(0)

    # If using captured data
    if use_capture == True:
        #ld.read_pcap_data('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-13-18-08_Velodyne-VLP-16-Data.pcap')   # Redo lidar_data
        
        # load data
        raw_lidar_data = np.load('./lidar_driver/lidar_trash_point_array.npy', allow_pickle=True)
        video = cv2.VideoCapture(video_path)
        model = torch.hub.load('camera_driver/yolov7', 'custom', path_or_model = 'camera_driver/yolo7.pt', source='local')
        #model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov7', 'custom', path_or_model = 'C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolo7.pt', source='local')
        #model = torch.hub.load('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolov7', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/camera_driver/yolo7_light.pt', source='local', force_reload=False)
        pos_stream = '../data/testrecord.txt'
        gps_date = sf.get_gps_date_ts(pos_stream)  # Need to resolve once

        # cast generators
        if ros == True:
            lidar_generator = ld.get_raw_lidar_data(bag, start_stamp, ros)
            camera_generator = cd.get_camera_frame(bag, start_stamp, ros=True) 
            pos_generator = sf.get_position(bag, start_stamp, relative_pos, ros=True)
        else:
            lidar_generator = ld.get_raw_lidar_data(raw_lidar_data, start_stamp)
            camera_generator = cd.get_camera_frame(video, start_stamp, video_path) 
            pos_generator = sf.get_position(pos_stream, start_stamp, relative_pos, gps_date)

        # Initialize data frames
        curr_lidar = next(lidar_generator)
        curr_cam = next(camera_generator)
        curr_pos = next(pos_generator)
        
        #Track initialization 
        last_position = curr_pos[1]
        measurement_model_cam = LinearGaussian(ndim_state=4, mapping=[0,2], noise_covar=np.diag([0.1**2, 0.1**2])) #0.2
        measurement_model_lid = LinearGaussian(ndim_state=4, mapping=[0,2], noise_covar=np.diag([0.1**2, 0.1**2])) #0.1
    while(runtime > ts):

        t1_start = process_time() 
        tracker_data = set()
        data_type, ts, data, curr_lidar, curr_cam, curr_pos = sf.data_handler(curr_lidar, curr_cam, curr_pos, lidar_generator, camera_generator, pos_generator)
        
        if data_type == 'lid':
            lidar_measurements, current_lines, thresholded_raw = ld.get_lidar_measurements(detector, data, position_delta = position_delta, radius = 10, intensity=0, heigth=-0.80, current_lines=current_lines)         # All lidar points on the water surface, bounds for plotting
            if np.size(current_lines) != 0:
                current_lines[:,1] = sf.get_relative_pos(current_lines[:,1], 'line')
                current_lines[:,1] = ld.update_lines_pos(0, current_lines[:,1])
            lm_plot = lidar_measurements
            data = ld.set_lidar_offset(+last_position[2], copy.deepcopy(lidar_measurements), t = [0.0+last_position[0], 0.0 +last_position[1]]) # -0.27 for old 021 new
            if np.size(data) != 0:
                for meas in data:
                    lidar_buffer.append(meas)
                    
            if buffer_index == 3:  # Buffer lidar measurements and cluster after
                clustered = ld.cluster_measurements(lidar_buffer)
                for c in clustered:
                    print(ts, "CLUSTER:", c)
                    ned_track.append([c, 'lid', ts])
                    tracker_data.add(Detection(state_vector =StateVector([c[1],c[0]]), timestamp =datetime.fromtimestamp(ts), measurement_model = measurement_model_lid))

                lidar_buffer = []
                buffer_index = 0

            else:
                buffer_index += 1
                t1_stop = process_time()
                print(data_type, ' : ', t1_stop-t1_start, ' : ', ts)
                continue
            
        elif data_type == 'cam':
            detections = cd.detect_trash(data, model, [1.5,0.0,-2.7]) 
            data = cd.set_cam_offset(last_position[2], detections,[0.0+last_position[0],-0.00+last_position[1]]) # -0.04 for old 045 new
            if np.size(data) != 0:
                for meas in data:
                    print(ts, "CAM DETECTION:", meas)
                    ned_track.append([meas,'cam', ts])
                    tracker_data.add(Detection(state_vector = StateVector([meas[1],meas[0]]), timestamp =datetime.fromtimestamp(ts), measurement_model = measurement_model_cam))

        elif data_type == 'pos':
            position_delta = np.array(data) - np.array(last_position)
            last_position = data
            track.append(last_position)
            curr_track = np.copy(track)
            tracker_data.add(Detection(state_vector = StateVector([last_position[0],last_position[1]]), timestamp =datetime.fromtimestamp(ts), measurement_model = measurement_model_lid)) #for plotting of gnss

            continue
        t1_stop = process_time()
        
        print(data_type, ' : ', t1_stop-t1_start, ' : ', ts)
        plot_data = [data_type, [], [], [], [], curr_track, [], np.copy(ned_track)]
        #plot_data = [data_type, thresholded_raw, lm_plot,current_lines, detections, curr_track, curr_cam[1], np.copy(ned_track)]
        
        yield ts, tracker_data, data_type, plot_data

def detector_wrapper(gen):
    for ts, tracker_data, _, _ in gen:
        yield datetime.fromtimestamp(ts), tracker_data
        
def main():

    start_stamp = 1675168544   
    runtime = start_stamp + int(input("Runtime (s): "))
    gt =  '../data/test_marcus/gnns_data/pbox.gpx'
    animation_data = []
    jpda = True
    tracks = set()
    

    if jpda == False:
        runner = run(start_stamp, runtime, ros = True, relative_pos = True)
        final_plot_data = None
        for ts, data, data_type, plot_data in runner:
                animation_data.append(plot_data)
                final_plot_data = plot_data
        RMSE_lid, RMSE_cam = sf.RMSE(final_plot_data[7], gt)
        print("RMSE lid :",RMSE_lid, "RMSE cam", RMSE_cam)
                
        pd.animate(animation_data)     
        
    else:
        runner = run(start_stamp, runtime, ros=False, relative_pos = False)
        gen = detector_wrapper(runner)
        tracker = jd.track(gen)
        for _, ctracks in tracker:
            tracks.update(ctracks)
        plotter = Plotterly()

        pil_img = Image.open("brattor_farge_korrekt.png").transpose(Image.FLIP_TOP_BOTTOM )
        prefix = "data:image/png;base64,"
        with BytesIO() as stream:
            pil_img.save(stream, format="png")
            base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
        
        
        #pil_img_usv = Image.open("usv.png").transpose(Image.FLIP_TOP_BOTTOM)
        #prefix = "data:image/png;base64,"
        #with BytesIO() as stream:
        #    pil_img_usv.save(stream, format="png")
        #    base64_string_usv = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
            
        #pos = [ 63.43998448192789* (np.pi/180),   10.399792069583912* (np.pi/180) ]   # For single sensor
        #start_pos = [63.4386345* (np.pi/180), 10.3985848* (np.pi/180)]  #brattør_farge 
        #ell_grs80 = pymap3d.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.31414036)
        #ned = pymap3d.geodetic2ned(pos[0], pos[1], 0, start_pos[0], start_pos[1], 0, ell=ell_grs80, deg=False)
    

        #test = plotly.graph_objects.Image(source=base64_string_usv, dx = 0.05, dy =0.05, x0 = ned[1]-1.35, y0 = ned[0]-1.4)
        
        plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
        plotter.fig.add_traces([plotly.graph_objects.Image(source=base64_string, dx = 0.295, dy = 0.265)]) #0.267
        plotter.fig["layout"]["yaxis"]["autorange"]=False
        plotter.fig.show()
        
        '''
        print("TRACKS", tracks)
        NIS, NEES, RMSE = sf.NIS_NEES_RMSE(tracks, gt)
        print(np.shape(np.array(NIS)))
        print(np.shape(np.array(NEES)))
        print(np.shape(np.array(RMSE)))

        for i, track in enumerate(NIS):
            pd.plot_nis(np.array(track)[:,0], np.array(track)[:,1], i)
        
        #try:    
        #    for i, track in enumerate(NEES):
        #        pd.plot_nees(np.array(track)[:,0], np.array(track)[:,1], i)
        #except Exception as E:
        #    print(E)
        #    pass
        try:
            for i, RM in enumerate(RMSE):
                print(RM)
        except:
            pass 
        '''

main()  
    

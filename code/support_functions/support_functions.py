import numpy as np
import pymap3d
from datetime import datetime, timedelta

import gpxpy
import gpxpy.gpx

def get_relative_pos(object, obj_t = 'point'):
    if obj_t == 'point':
        if np.size(object) == 0:
            return np.array([])
        new_obj = np.empty_like(object)
        new_obj = (object / 5)- 10
        
        return new_obj
    
    elif obj_t == 'line':
        relative_lines = []
        if np.size(object) == 0:
            return []
        for line in object:
            x0 = (line[0] / 5)- 10
            y0 = (line[1] / 5)- 10
            x1 = (line[2] / 5)- 10
            y1 = (line[3] / 5) -10
            rel_line = [-x0, y0, -x1, y1]
            relative_lines.append(rel_line)           
        return relative_lines
    
    return None

def get_image_pos(elem):
    
    image_elem = (np.floor((elem+10)*5)).astype(int)
    
    return image_elem

def get_gps_date_ts(data_stream):
    file = open(data_stream, 'r')
    frames = file.readlines()
    for frame in frames:
        if frame[1:6] == 'GNRMC':
            date = frame.split(',')[9]
            date= datetime.strptime(date, "%d%m%y")
            return date

def get_position(data_stream, start_stamp, relative_pos, date = None, ros = False):

    set_start_pos = False
    heading = 0
    if ros == True:
        if relative_pos == True:
            t = start_stamp
            for i in range(1000):
                t +=1
                yield [t, [0,0,227]]
                
        else:
            #for final test
            for topic, msg, t in data_stream.read_messages(topics=['/velodyne_packets']):
                t = t.to_sec()
                if start_stamp > t:
                            continue
                
                if set_start_pos == False:
                    start_pos = [63.4386345* (np.pi/180), 10.3985848* (np.pi/180)]
                    set_start_pos = True
                    
                lat = 63.440001* (np.pi/180)
                lon = 10.399845* (np.pi/180) 
                
                ell_grs80 = pymap3d.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.31414036)
                ned = pymap3d.geodetic2ned(lat, lon, 0, start_pos[0], start_pos[1], 0, ell=ell_grs80, deg=False)
                
                yield [t, [ned[0], ned[1], 227]]
            
    else:
        file = open(data_stream, 'r')
        frames = file.readlines()
        for i, frame in enumerate(frames):
            if frame[1:6] == 'GPGGA':
                pos_data = frame.split(',')
                time   = pos_data[1].replace('.', '')+'0000'
                lat = (float(pos_data[2][0:2]) + float(pos_data[2][2:]) /60) * (np.pi/180)
                lon = (float(pos_data[4][0:3]) + float(pos_data[4][3:]) /60) * (np.pi/180)
                j, found = 1, False
                while frames[i+j][1:6] != 'GPGGA' and j < 10 and found == False:
                    if frames[i+j][1:6] == 'GPHDT':
                        heading=float(frames[i+j].split(',')[1])
                        found = True
                    j += j
                curr_time = datetime.strptime(time, "%H%M%S%f")
                delta = timedelta(hours = curr_time.hour+1, minutes =curr_time.minute, seconds = curr_time.second, milliseconds= curr_time.microsecond*1/1000)
                curr_date = date + delta
                timestamp = datetime.timestamp(curr_date)
                
                if start_stamp > timestamp:
                    continue
                if set_start_pos == False:
                    if relative_pos == True:
                        start_pos= [float(lat), float(lon)]
                    else:
                        start_pos = [63.4386345* (np.pi/180), 10.3985848* (np.pi/180)]  #brattør_farge 
                    set_start_pos = True
                ell_grs80 = pymap3d.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.31414036)
                ned = pymap3d.geodetic2ned(lat, lon, 0, start_pos[0], start_pos[1], 0, ell=ell_grs80, deg=False)
                yield [timestamp, [ned[0], ned[1], heading]]
            

def data_handler(curr_lidar, curr_cam, curr_pos, gen_lidar, gen_cam, gen_pos):
    data_type = ''
    data = None
    if curr_lidar[0] < curr_cam[0] and curr_lidar[0] < curr_pos[0]:
        data_type = 'lid'
        ts =  curr_lidar[0]
        data = curr_lidar[1]
        curr_lidar = next(gen_lidar)
    elif curr_cam[0] < curr_lidar[0] and curr_cam[0] < curr_pos[0]:
        data_type = 'cam'
        ts =  curr_cam[0]
        data = curr_cam[1]
        curr_cam = next(gen_cam)
    else:
        data_type = 'pos'
        ts =  curr_pos[0]
        data = curr_pos[1]
        curr_pos = next(gen_pos)
        
    return data_type, ts, data, curr_lidar, curr_cam, curr_pos

def rotation_matrix(psi, theta, phi):
    
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                  [np.sin(psi), np.cos(psi), 0.0],
                  [0.0,0.0,1.0]])
    R_y = np.array([[np.cos(theta),0.0, np.sin(theta)],
                  [0.0,1.0,0.0],
                  [-np.sin(theta),0.0, np.cos(theta)]])
    R_x = np.array([[1.0,0.0,0.0],
                    [0.0,np.cos(phi), -np.sin(phi)],
                    [0.0,np.sin(phi), np.cos(phi)]])
    R = (R_z@R_y@R_x).round(5)
    
    return R

def NIS_NEES_RMSE(tracker, gt):
    track_NIS = []
    track_NEES = []
    track_RMSE = []
    
    for track in tracker.tracks:
        start_ts = datetime.timestamp(track[0]._property_timestamp)
        gt_gen = gt_generator(gt, start_ts)
        ned_gt = next(gt_gen)
        print('--------------- NEW TRACK ---------------')
        print(start_ts, ned_gt)
        NIS = []
        NEES = []
        RMSE = []
        for meas in track:
            
            ts = datetime.timestamp(meas._property_timestamp)

            
            #NEES extraction
            x_hat = meas._property_state_vector
            P = meas._property_covar
            P_pos = np.array([[P[0,0],P[0,3]],[P[3,0], P [3,3]]])
            
            if ts + 0.2 >= ned_gt[0] >= ts -0.2:
                x_err = np.array([x_hat[0], x_hat[2]]).T-np.array(ned_gt[1]).T
                print(x_err)
                NEES.append(get_NEES(x_err, P_pos))
                RMSE.append(get_RMSE(x_err))
                
                ned_gt = next(gt_gen)
            elif ned_gt[0] < ts - 0.5:
                ned_gt = next(gt_gen)
            
            #NIS extraction
            try:
                if meas._property_hypothesis._property_measurement_prediction != None:
                    y = meas._property_hypothesis._property_measurement._property_state_vector
                    y_hat = meas._property_hypothesis._property_measurement_prediction._property_state_vector
                    S = meas._property_hypothesis._property_measurement_prediction._property_covar
                    NIS.append([ts, get_NIS(y, y_hat, S)])
            except Exception as E:  
                try:
                    if meas._property_hypothesis._property_single_hypotheses != None:
                        for hyp in meas._property_hypothesis._property_single_hypotheses:
                            if hyp._property_probability >= 0.95:                            
                                y = hyp._property_measurement._property_state_vector
                                y_hat = hyp._property_measurement_prediction._property_state_vector
                                S = hyp._property_measurement_prediction._property_covar
                                NIS.append([ts, get_NIS(y, y_hat, S)])
                                
                except Exception as E:  
                    pass
        track_NIS.append(NIS)
        track_NEES.append(NEES)
        track_RMSE.append(RMSE)
        
    print("NIS",track_NIS)
    print("NEES",track_NEES)
    print("RMSE",track_RMSE)
    return track_NIS, track_NEES, track_RMSE

def get_NIS(meas, pred_meas, meas_cov):
    NIS = (pred_meas - meas).T @ np.linalg.inv(meas_cov) @ (pred_meas - meas)
    return NIS[0]

def get_RMSE(error):

     return np.sqrt(np.mean((error)**2))

def get_NEES(error, P_k):
    P_k_inv = np.linalg.inv(P_k)
    NEES =  np.sum(error.T @ P_k_inv @ error)
    return NEES

def gt_generator(gpx, start_stamp):
    gpx_file = open(gpx, 'r')
    gpx = gpxpy.parse(gpx_file)
    ned_old = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                timestamp = datetime.timestamp(point.time )
                start_pos = [63.4386345* (np.pi/180), 10.3985848* (np.pi/180)]  #brattør_farge 
                ell_grs80 = pymap3d.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.31414036)
                ned = pymap3d.geodetic2ned(point.latitude* (np.pi/180), point.longitude* (np.pi/180), 0, start_pos[0], start_pos[1], 0, ell=ell_grs80, deg=False)
                if start_stamp > timestamp:
                   ned_old = ned
                   continue
               
                if ned[1] == ned_old[1] and ned[0] == ned_old[0]: # No new measurement
                    continue
                ned_old = ned
                yield [timestamp, [ned[1], ned[0]]]
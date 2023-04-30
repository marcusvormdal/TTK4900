import numpy as np
import pymap3d
from datetime import datetime, timedelta
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

def get_position(data_stream, date, start_stamp):
    file = open(data_stream, 'r')
    frames = file.readlines()
    set_start_pos = False
    heading = 0
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
                start_pos= [float(lat), float(lon)]
                set_start_pos = True
            ned = pymap3d.geodetic2ned(lat, lon, 0, start_pos[0], start_pos[1], 0, ell=None, deg=False)
            yield [timestamp, [ned[0], ned[1], heading]]
            

def data_handler(curr_lidar, curr_cam, curr_pos, gen_lidar, gen_cam, gen_pos):
    #print('Lidar_t', curr_lidar[0])
    #print('Camera_t', curr_cam[0])
    #print('Position_t', curr_pos[0])
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


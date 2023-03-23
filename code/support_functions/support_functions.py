import numpy as np
import pymap3d
def get_relative_pos(object, obj_t = 'point', centered = False):
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
            rel_line = [(y0, y1),(x0, x1)]

            relative_lines.append(rel_line)           
        return relative_lines
    
    return None

def get_image_pos(elem):
    
    image_elem = (np.floor((elem+10)*5)).astype(int)
    
    return image_elem

def get_position(data_stream, start_frame = 0, start_pos = None):
        
    print(start_pos)
    file = open(data_stream, 'r')
    frames = file.readlines()
    for frame in frames:
        if frame[1:6] == 'GPGGA':
            
            pos_data = frame.split(',')
            time   = float(pos_data[1])
            lat = (float(pos_data[2][0:2]) + float(pos_data[2][2:]) /60) * (np.pi/180)
            lon = (float(pos_data[4][0:3]) + float(pos_data[4][3:]) /60) * (np.pi/180)
            speed, theta =  float(pos_data[6]), float(pos_data[7])*np.pi/180
            if start_pos == None:
                start_pos= [float(lat), float(lon)]
            #print("Start", start_pos)
            #print(time, lat, lon, speed, theta)

            ned = pymap3d.geodetic2ned(lat, lon, 0, start_pos[0], start_pos[1], 0, ell=None, deg=False)
            print("NED",ned)
            yield [time, ned[0], ned[1], theta]

def update_position(position_delta, element):
    
    R = np.array([[np.cos(-position_delta[3]), -np.sin(-position_delta[3]), 0],
                  [np.sin(-position_delta[3]), np.cos(-position_delta[3]), 0],
                  [0,0,1]])
    new_elem = R @ np.array([element[0], element[1], 0]) - position_delta[1:3]
    
    return new_elem[0:2]

def data_handler(curr_lidar, curr_cam, curr_pos, gen_lidar, gen_cam, gen_pos):
    data_type = ''
    data = None
    if curr_lidar.t <curr_cam.t and curr_lidar.t <curr_pos.t:
        data_type = 'lidar'
        data = curr_lidar
        curr_lidar = next(gen_pos)
    if curr_cam.t <curr_lidar.t and curr_cam.t <curr_pos.t:
        data_type = 'cam'
        data = curr_cam
        curr_cam = next(gen_lidar)
    else:
        data_type = 'pos'
        data = curr_pos
        curr_pos = next(gen_cam)
    return data_type, data, curr_lidar, curr_cam, curr_pos
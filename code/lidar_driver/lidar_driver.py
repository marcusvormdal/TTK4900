import numpy as np
import velodyne_decoder as vd
import cv2
from PIL import Image as im
from support_functions.support_functions import get_image_pos
def read_pcap_data(filepath):
    config = vd.Config(model='VLP-16', rpm=600)
    cloud_arrays = []
    for _, points in vd.read_pcap(filepath, config):
        cloud_arrays.append(points)

    np.save('lidar_trash_point_array.npy', np.array(cloud_arrays, dtype=object))
    return

def get_raw_lidar_data(raw_lidar_data, start_frame):
    for frame in raw_lidar_data[start_frame:]:
        yield frame
        
def get_lidar_measurements(detector, lidar_data, position_delta, radius, intensity, heigth, current_lines):
    if np.size(lidar_data) == 0:
        return None, None, None, None
    frame_points = []
    unique_points = np.unique(lidar_data, axis=0)    
    
    for point in unique_points:
        if np.linalg.norm([point[0], point[1]])<= radius and point[2] < heigth and point[3] > intensity: 
            frame_points.append(point)
            
    frame_points = np.array(frame_points)
    
    lines = []
    new_lines = []
    lidar_measurements = []
    if np.size(frame_points) != 0:
        lidar_image = lidar_to_image(frame_points)
        #cv2.imshow('converted', lidar_image)
        lidar_image = cv2.GaussianBlur(lidar_image,(3,3),0)
        new_lines = detector.detect(lidar_image)
        lines = update_lines(new_lines, current_lines, position_delta)
        lidar_measurements = clean_on_line_intersect(lines, frame_points)
    
    return lidar_measurements, lines, frame_points, new_lines


def lidar_to_image(lidar_points, height = 100, width = 100):   
    x = get_image_pos(lidar_points[:,0])
    y = get_image_pos(lidar_points[:,1])
    img_numpy = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(np.shape(x)[0]):
        img_numpy[x[i], y[i]] = 255
        
    return img_numpy


def clean_on_line_intersect(lines, lidar_points):
    if np.size(lines) == 0:
        return lidar_points

    lines = lines[:,1]

    cleaned_points = []
    x = get_image_pos(lidar_points[:,0])
    y = get_image_pos(lidar_points[:,1])
    for i in range(np.size(x)):
        intersect = False
        y1, x1, y2, x2 = 50,50, x[i], y[i]
        
        for l in lines:
            x3, y3, x4, y4 = np.floor(l[0]), np.floor(l[1]), np.floor(l[2]), np.floor(l[3])

            t_num = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
            t_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            u_num = (x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)
            u_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if u_den == 0 or t_den == 0:
                continue
            t = t_num / t_den
            u = u_num /u_den
            if t >= 0.5:
                t = t - 0.2
            else:
                t = t + 0.2
            if u >= 0.5:
                u = u - 0.2
            else:
                u = u + 0.2
            
            if (0 <= t <= 1) and (0 <= u <= 1):
                intersect = True
                
        if not intersect:  
           cleaned_points.append(lidar_points[i])
    
    return np.array(cleaned_points)
    
def update_lines(lines, current_lines, position_delta):
    
    current_lines = update_lines_pos(position_delta, current_lines)
    current_lines = list(current_lines)
    current_lines = remove_outdated_lines(current_lines)
    
    if np.size(lines) > 0 and lines[0] != None:
        for l in lines[0]:
            if np.size(current_lines) == 0:
                current_lines = [[0,l[0]]]
            else:
                current_lines.append([0, l[0]])

    return np.array(current_lines, dtype=object)

def remove_outdated_lines(lines):
    updated_lines = []
    for l in lines:
        if l[0] != 2:
            l[0] += 1
            updated_lines.append(l)
    return updated_lines

def update_lines_pos(position_delta, lines):
    updated_lines = []
    R = np.array([[np.cos(-position_delta[3]), -np.sin(-position_delta[3]), 0],
                [np.sin(-position_delta[3]), np.cos(-position_delta[3]), 0],
                [0,0,1]])
    print("delta", position_delta)
    for l in lines:
        #print("Before",l)
        new_line_start = (R @ np.array([l[1][0], l[1][1], 0]))[0:2] - position_delta[1:3]
        new_line_end = (R @ np.array([l[1][2], l[1][3], 0]))[0:2] - position_delta[1:3]
        l[1][0], l[1][1] = new_line_start[0], new_line_start[1]
        l[1][2], l[1][3] = new_line_end[0], new_line_end[1]
        #print("After",l)
        updated_lines.append(l)
    return updated_lines
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
        
def get_lidar_measurements(detector, lidar_data, radius, intensity, heigth, ):
    frame_points = []
    unique_points = np.unique(lidar_data, axis=0)    
    
    for point in unique_points:
        if np.linalg.norm([point[0], point[1]])<= radius and point[2] < heigth and point[3] > intensity: 
            frame_points.append(point)
            
    frame_points = np.array(frame_points)
    
    lines = []
    lidar_measurements = []
    if np.size(frame_points) != 0:
        lidar_image = lidar_to_image(frame_points)
        lidar_image = cv2.GaussianBlur(lidar_image,(3,3),0)
        lines = detector.detect(lidar_image)
        lidar_measurements = clean_on_line_intersect(lines[0], frame_points)
    
    return lidar_measurements, lines, frame_points


def lidar_to_image(lidar_points, height = 100, width = 100):   
    x = get_image_pos(lidar_points[:,0])
    y = get_image_pos(lidar_points[:,1])
    img_numpy = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(np.shape(x)[0]):
        img_numpy[x[i], y[i]] = 255
        
    return img_numpy


def clean_on_line_intersect(lines, lidar_points):
    cleaned_points = []
    x = get_image_pos(lidar_points[:,0])
    y = get_image_pos(lidar_points[:,1])

    for i in range(np.size(x)):
        intersect = False
        y1, x1, y2, x2 = 50,50, x[i], y[i]
        
        for l in lines:
            x3, y3, x4, y4 = np.floor(l[0][0]), np.floor(l[0][1]), np.floor(l[0][2]), np.floor(l[0][3])
            #x3, y3, x4, y4 = pad_line(np.floor(l[0][0]), np.floor(l[0][1]), np.floor(l[0][2]), np.floor(l[0][3]))

            t_num = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
            t_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            u_num = (x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)
            u_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            t = t_num / t_den
            u = u_num /u_den
            if (0 <= t <= 1) and (0 <= u <= 1):
                intersect = True
                
        if not intersect:  
           cleaned_points.append(lidar_points[i])
    
    return np.array(cleaned_points)

def pad_line(x1, y1, x2, y2):


    return x1, y1, x2, y2 


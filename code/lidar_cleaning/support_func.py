import numpy as np
from PIL import Image as im
import cv2

def lidar_to_image(lidar_points, height = 100, width = 100):
    # get coordinates in cm from range (0-2000), i.e. centre is (1000, 1000)
    
    x = (np.floor((lidar_points[:,0]+10)*5)).astype(int)
    y = (np.floor((lidar_points[:,1]+10)*5)).astype(int)
    img_numpy = np.zeros((height, width), dtype=np.uint8)
    for i in range(np.shape(x)[0]):
        img_numpy[x[i], y[i]] = 255
    return img_numpy

def clean_on_line_intersect(lines, points):
    cleaned_points = []
    #centered_points = get_image_pos(points, 'point', centered = True)
    x = (np.floor((points[:,0]+10)*5)).astype(int)
    y = (np.floor((points[:,1]+10)*5)).astype(int)

    for i in range(np.size(x)):
        intersect = False
        x1, y1, x2, y2 = 50,50, x[i], y[i]
        if 200< i < 300:
            print(lines)
            print(x1, y1, x2, y2)
        for l in lines:
            x3, y3, x4, y4 = l[0][0], l[0][1], l[0][2], l[0][3]
            t_num = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
            t_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            u_num = (x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)
            u_den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            t = t_num /t_den
            u = u_num /u_den
            if (0 <= t <= 1) or (0 <= u <= 1):
                intersect = True
        if not intersect:                     
            cleaned_points.append(points[i])
    return np.array(cleaned_points)


def get_relative_pos(object, obj_t = 'point', centered = False):
    offset_x = 0
    offset_y = 0
    
    if centered == True:
        offset_x, offset_y = 50, 50
        
    if obj_t == 'point':
        if np.size(object) == 0:
            return np.array([])
        new_obj = np.empty_like(object)
        new_obj = (object / 5)- 10
        
        return new_obj
    
    elif obj_t == 'line':
        relative_lines = []

        for line in object[0]:
            x0 = (line[0][0] / 5)- 10
            y0 = (line[0][1] / 5)- 10
            x1 = (line[0][2] / 5)- 10
            y1 = (line[0][3] / 5) -10
            rel_line = [(y0, y1),(x0, x1)]

            relative_lines.append(rel_line)           
        return relative_lines
    
    return None

def get_image_pos(object, obj_t, centered = False):

    if obj_t == 'point':
        new_obj = np.empty_like(object)
        new_obj[:,0] = (np.floor((object[:,0]+10)*5)).astype(int)
        new_obj[:,1] = (np.floor((object[:,1]+10)*5)).astype(int)   
        
        return new_obj
        
    elif obj_t == 'lines':
        
        return None
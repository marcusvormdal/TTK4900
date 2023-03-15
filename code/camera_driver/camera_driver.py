import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd 


def get_camera_frame(video, start_frame):
    
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    success = True
    while success:
        success,img = video.read()
        img = cv2.convertScaleAbs(img, alpha=1.7, beta=0)
        
        yield img
        
def detect_trash(image, model):

    predictions = model(image)
        
    boxes = []
    detections = []
    for row in predictions.pandas().xyxy[0].itertuples():
        if row.confidence > 0.35 and row.ymin > 725 and row.ymax > 725:
            print("detect",row)
            if (row.ymin > 1200 or row.ymax > 1200) and (row.xmin > 700 or row.xmax > 700) and (row.xmin < 2100 or row.xmax < 2100):  # filter boat front
                continue
            detections.append(row)
            box = calculate_angle(row)
            boxes.append(box)
            #print(np.rad2deg(box[0])-90)
    print(detections)
    for b in boxes:
        world_cord = get_world_coordinate(0,0,0,0.63, b)
    
    return detections, boxes


def calculate_angle(bbox):
    """Assumes plane on water level"""
    obj_point = ((bbox.xmax+bbox.xmin)/2, bbox.ymax)  # ymax to do bottom detection
    box_data = (np.arctan2(1520-obj_point[1], obj_point[0]-1344)*180/np.pi-90, obj_point[0], obj_point[1])
    print(box_data)
    return box_data


def get_world_coordinate(theta, t1,t2,t3, box_coord):
    f = 80
    ox = 1344
    oy = 760
    
    A = np.array([[np.cos(theta) / f , np.sin(theta) / f, (-1/(f*t3))*(np.cos(theta)*f*t1+np.sin(theta)*f*t2 + np.cos(theta)*ox*t3+ np.sin(theta)*oy*t3) ],
                  [-np.sin(theta) / f , np.cos(theta) / f, (-1/(f*t3))*(np.cos(theta)*f*t2-np.sin(theta)*f*t1 + np.cos(theta)*oy*t3- np.sin(theta)*ox*t3) ],
                  [0,0,1/t3],])
    hom_cord = np.array([box_coord[1], box_coord[2], t3])
    world_cord = A @ hom_cord.T
    print("Wcord:",world_cord, np.arctan2(world_cord[1], world_cord[0])*180/np.pi)
    return world_cord
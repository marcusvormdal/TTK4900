import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd 

def get_camera_frame(video, start_frame):
    
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    success = True
    while success:
        success,img = video.read()
        
        yield img
        
def detect_trash(image, model):
    prediction = model(image)    
    #print(prediction.pandas().xyxy[0])
    boxes = []
    for row in prediction.pandas().xyxy[0].itertuples():
        if row.confidence > 0.45 and row.ymin > 514 and row.ymax > 514:
            box = calculate_angle(row)
            boxes.append(box)
            #print(np.rad2deg(box[0])-90)
            
    return prediction, boxes

def get_image_measurements():
    
    return None


def calculate_angle(bbox):
    """Assumes plane on water level"""
    image_centre =(1344, 760)
    obj_centre = ((bbox.xmax+bbox.xmin)/2, (bbox.ymax+bbox.ymin)/2)
    box_data = (np.arctan2(1520-obj_centre[1], image_centre[0]-obj_centre[0],), obj_centre[0], obj_centre[1])
    return box_data



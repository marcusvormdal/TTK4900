import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd 

vidcap = cv2.VideoCapture('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/videos/trash_collect.mp4')
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 1800)

success,img = vidcap.read()

model = torch.hub.load('yolov5', 'custom', path ='C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/yolo5_model/pLitterFloat_800x752_to_640x640.pt', source='local', force_reload=True)
window_name = 'image'

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame+50)

def calculate_angle(bbox):
    """Assumes plane on water level"""
    image_centre =(1344, 760)
    obj_centre = ((bbox.xmax+bbox.xmin)/2, (bbox.ymax+bbox.ymin)/2)
    box_data = (np.arctan2(1520-obj_centre[1], image_centre[0]-obj_centre[0],), obj_centre[0], obj_centre[1])
    return box_data

wait = False
boxes = []

while success:
    success,img = vidcap.read()
    prediction = model(img)    
    print(prediction.pandas().xyxy[0])
    
    for row in prediction.pandas().xyxy[0].itertuples():
        if row.confidence > 0.45 and row.ymin > 514 and row.ymax > 514:
            box = calculate_angle(row)
            print(np.rad2deg(box[0])-90)
            cv2.line(img, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
            wait = True

    print('Read a new frame: ', success)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    imS = cv2.resize(img, (1920, 1080))                # Resize image
    
    if wait == True:
        cv2.line(imS, (960, 1080), (960, 800), (0, 0, 255), 2)
        cv2.imshow("output", imS)
        cv2.waitKey(0)                      
        wait = False
        boxes = []
    else:    
        cv2.imshow("output", imS)               

    cv2.waitKey(250)
    cv2.setMouseCallback('output', click_event)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import ffmpeg
from pprint import pprint
from datetime import datetime, timedelta
from time import process_time


def get_camera_frame(video, start_stamp, video_path):
    meta_data = ffmpeg.probe(video_path)
    end_time = datetime.fromisoformat(meta_data['format']['tags']['creation_time'][0:19])
    duration = meta_data['streams'][0]['duration']
    frame_num = int(meta_data['streams'][0]['nb_frames'])
    m = timedelta(minutes=(int(float(duration)/60))-60)
    s = timedelta(seconds=(int(float(duration)%60)+3))
    start_time = end_time - m - s
    stamp = datetime.timestamp(start_time) - 0.25 
    video_offset = int((start_stamp - stamp) * 4 - 15)
    stamp = stamp + int(start_stamp - stamp) -15
    video.set(cv2.CAP_PROP_POS_FRAMES, video_offset)
    success = True
    for i in range(frame_num):
        success, img = video.read()
        stamp = stamp + 0.25
        if start_stamp > stamp or not success:
            continue
        img = cv2.convertScaleAbs(img, alpha=1.7, beta=0)
        yield [stamp, img]

def detect_trash(image, model):
    t1_start = process_time() 
    predictions = model(image)
    t1_stop = process_time()
    print('cam processing : ', t1_stop-t1_start)
    boxes = []
    detections = []
    for row in predictions.pandas().xyxy[0].itertuples():
        if row.confidence > 0.35 and row.ymin > 725 and row.ymax > 725:
            #print("detect",row)
            if (row.ymin > 1200 or row.ymax > 1200) and (row.xmin > 700 or row.xmax > 700) and (row.xmin < 2100 or row.xmax < 2100):  # filter boat front
                continue
            detections.append(row)
            box = calculate_angle(row)
            boxes.append(box)
            #print(np.rad2deg(box[0])-90)
    if np.size(detections) > 0:
        print("Detections:", detections)
    for b in boxes:
        world_cord = get_world_coordinate(0,-90*np.pi/180,0,0,0,0.63, b)
    
    return detections, boxes


def calculate_angle(bbox):
    """Assumes plane on water level"""
    obj_point = ((bbox.xmax+bbox.xmin)/2, bbox.ymax)  # ymax to do bottom detection
    box_data = (np.arctan2(1520-obj_point[1], obj_point[0]-1344)*180/np.pi-90, obj_point[0], obj_point[1])
    #print(box_data)
    return box_data


def get_world_coordinate(psi,theta,phi, t1,t2,t3, box_coord):
    f = 2.8
    ox = 1344
    oy = 760
    
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                  [np.sin(psi), np.cos(psi), 0],
                  [0,0,1]])
    R_y = np.array([[np.cos(theta),0, np.sin(theta)],
                  [0,1,0],
                  [-np.sin(theta),0, np.cos(theta)]])
    R_x = np.array([[1,0,0],
                    [0,np.cos(phi), -np.sin(phi)],
                    [0,np.sin(phi), np.cos(phi)]])
    
    R_bc = (R_z@R_y@R_x)
    R_bc_inv = np.linalg.inv(R_bc).round(5)
    R = np.zeros([3,3])
    
    R[0:2, 0:2] = R_bc[0:2, 0:2]

    test = [t1,t2,t3]
    test = -R_bc_inv @ test 
    
    print(test)
    
    R[2,0], R[2,1], R[2,2] = test[0], test[1], test[2]
    i = np.array([[f, 0, ox],[0,f, oy],[0,0,1]])

    print(np.linalg.pinv(R).round(5))
    print(np.linalg.inv(i).round(5))
    hom_cord = np.array([box_coord[1], box_coord[2], 1])
    world_cord = (np.linalg.pinv(R).round(5) @ np.linalg.inv(i).round(5) @ hom_cord.T).round(2) / 1.59
    
    print("Wcord:",world_cord, np.arctan2(world_cord[1], world_cord[0])*180/np.pi)
    return world_cord

def test_world_coord():
    psi,theta,phi = (np.pi/180)*90,0,np.pi
    t1,t2,t3 = 0,0,0.63
    box_coord  = (0,1344,1090)
    
    world_cord = get_world_coordinate(psi, theta, phi, t1, t2, t3, box_coord)
    
def __main__():
    test_world_coord()
    
__main__()
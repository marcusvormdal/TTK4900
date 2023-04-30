import numpy as np
import cv2
import ffmpeg
from datetime import datetime, timedelta
#from support_functions.support_functions import rotation_matrix

def get_camera_frame(video, start_stamp, video_path):
    meta_data = ffmpeg.probe(video_path)
    end_time = datetime.fromisoformat(meta_data['format']['tags']['creation_time'][0:19])
    duration = meta_data['streams'][0]['duration']
    frame_num = int(meta_data['streams'][0]['nb_frames'])
    m = timedelta(minutes=(int(float(duration)/60))-60)
    s = timedelta(seconds=(int(float(duration)%60)+3))
    start_time = end_time - m - s
    stamp = datetime.timestamp(start_time) - 0.25 
    video_offset = int((start_stamp - stamp) * 4 - 6*4)+2
    stamp = stamp + int(start_stamp - stamp)
    video.set(cv2.CAP_PROP_POS_FRAMES, video_offset)
    success = True
    for i in range(frame_num):
        success, img = video.read()
        stamp = stamp + 0.25
        if start_stamp > stamp or not success:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.convertScaleAbs(img, alpha=1.7, beta=0)
        yield [stamp, img]

def detect_trash(image, model, rot):
    image_res = cv2.resize(image, (672, 380))       
    predictions = model(image_res)
    boxes = []
    detections = []
    world_coords = []
    for row in predictions.pandas().xyxy[0].itertuples():
        ymin, ymax = int(row.ymin *4), int(row.ymax *4)
        xmin, xmax = int(row.xmin *4), int(row.xmax *4)
        if row.confidence > 0.35 and ymin > 725 and ymax > 725:
            if (ymin > 1200 or ymax > 1200) and (xmin > 615 or xmax > 615) and (xmin < 2050 or xmax < 2050):  # filter boat front
                continue
            if (ymin > 1490 or ymax > 1490) and (xmin > 2550 or xmax > 2550):  # filter boat front
                continue
            detections.append([xmin, ymin,xmax, ymax])
            box = calculate_angle(ymax, xmin, xmax)
            boxes.append(box)
    #if np.size(detections) > 0:
        #print("Detections:", detections)
    for b in boxes:
        R = rotation_matrix(np.radians(-90.0+rot[0]), rot[1], np.radians(90.0+rot[2]))
        world_coord = alternate_world_coord(b[1],b[2], R, [0.0,0.0,0.99])
        if world_coord[0] < 8.0:
            world_coords.append(world_coord)
        
    return [detections, world_coords, boxes]


def calculate_angle(ymax, xmin, xmax):
    """Assumes plane on water level"""
    obj_point = ((xmax+xmin)/2, ymax)  # ymax to do bottom detection
    box_data = (np.arctan2(1520-obj_point[1], obj_point[0]-1344)*180/np.pi-90, obj_point[0], obj_point[1])
    #print(box_data)
    return box_data

def get_world_coordinate(psi,theta,phi, t1,t2,t3, box_coord):
    #world_cord = get_world_coordinate(np.radians(-90),np.radians(-90),0,0,0,-0.63, b)
    ox = 1344
    oy = 760
    R = rotation_matrix(psi,theta,phi)
    
    col_3 = -R @ [t1,t2,t3] 
    R[0,2], R[1,2], R[2,2] = col_3[0], col_3[1], col_3[2]
        
    i = np.array([[1307, 0, ox],[0,1307, oy],[0,0,1]])
    hom_cord = np.array([box_coord[1], box_coord[2], 1])
    
    world_cord = (np.linalg.inv(R) @ np.linalg.inv(i).round(5) @ hom_cord.T).round(3)
    
    #print("Wcord:",world_cord, np.arctan2(world_cord[1], world_cord[0])*180/np.pi)
    return world_cord

def test_world_coord():
    psi,theta,phi = np.radians(-85),0,np.radians(80)
    box_coord  = (0,131,1041)
    #world_cord = get_world_coordinate(psi, theta, phi, t1, t2, t3, box_coord)
    R = rotation_matrix(psi,theta,phi)
    world_cord_2 = alternate_world_coord(box_coord[1],box_coord[2], R, [0,0,0.63])
    

def alternate_world_coord(u,v, R, t_wc):
    theta = ((u - 1344) / 2688)*np.radians(109)
    psi =  ((v - 760) / 1520)*np.radians(60)
    v_c = [np.tan(theta), np.tan(psi), 1]
    v_w = R@v_c + np.array([t_wc[0], t_wc[1], 0])
    s = -t_wc[2] / v_w[2]
    x_w = t_wc + s*v_w
    #print("x_w",x_w)
    return x_w.round(2)

def rotation_matrix(psi, theta, phi):
    
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                  [np.sin(psi), np.cos(psi), 0],
                  [0,0,1]])
    R_y = np.array([[np.cos(theta),0, np.sin(theta)],
                  [0,1,0],
                  [-np.sin(theta),0, np.cos(theta)]])
    R_x = np.array([[1,0,0],
                    [0,np.cos(phi), -np.sin(phi)],
                    [0,np.sin(phi), np.cos(phi)]])
    R = (R_z@R_y@R_x).round(5)
    
    return R

def set_cam_offset(rot, detections,t):
    cam_measurements = []
    R = rotation_matrix(np.radians(rot),0,0)[0:2,0:2]
    if np.size(detections[1]) !=[]:
        for det in detections[1]:       
            measure = R @ np.array([det[0],det[1]]).T
            #print("bef",measure)
            #print("t",t)
            measure = measure.T + t
            #print("aft",measure)
            cam_measurements.append(measure)
    return cam_measurements

#def __main__():
#    test_world_coord()
    
#__main__()


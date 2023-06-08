import numpy as np
import cv2
import ffmpeg
from datetime import datetime, timedelta
from cv_bridge import CvBridge
import glob

bridge = CvBridge()

def get_camera_frame(video, start_stamp, video_path = None, ros = False):
    intrinsic = cv2.UMat(np.array([[2.75344274e+03, 0.00000000e+00, 1.34664016e+03],[0.00000000e+00, 2.77845260e+03, 7.48362894e+02], 
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])) 
    distortion = cv2.UMat(np.array([[-0.94911185,  2.27298045 , 0.02827832 ,-0.00913316 ,-3.70567064]]))
    
    if ros == True:
        t = 0
        for topic, msg, t in video.read_messages(topics=['/camera/image_raw/compressed']):
            t = t.to_sec()
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv2.undistort(cv_img, intrinsic, distortion, None, None)
            cv_img = cv2.UMat.get(cv_img)
            if start_stamp > t:
                continue
            t = t-25
            #cv2.imshow('img', cv_img)
            #cv2.waitKey(0)
            yield [t,cv_img]
            
        yield [t*4, None]  # If offset thorws off measurements
        
    else:
        meta_data = ffmpeg.probe(video_path)
        end_time = datetime.fromisoformat(meta_data['format']['tags']['creation_time'][0:19])
        duration = meta_data['streams'][0]['duration']
        frame_num = int(meta_data['streams'][0]['nb_frames'])
        m = timedelta(minutes=(int(float(duration)/60))-60)
        s = timedelta(seconds=(int(float(duration)%60)+3))
        start_time = end_time - m - s
        stamp = datetime.timestamp(start_time) - 0.25 
        video_offset = int((start_stamp - stamp) * 4 - 6*4)-7
        stamp = stamp + int(start_stamp - stamp)
        video.set(cv2.CAP_PROP_POS_FRAMES, video_offset)
        success = True
        for i in range(frame_num):
            success, img = video.read()
            stamp = stamp + 0.25
            if start_stamp > stamp or not success:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    for b in boxes:
        R = rotation_matrix(np.radians(-90.0+rot[0]), rot[1], np.radians(90.0+rot[2]))
        world_coord = georeference(b[1],b[2], R, [0.0,0.0,0.90])
        if world_coord[0] < 10.0:
            world_coords.append(world_coord)
    #Temp
    '''
    if np.size(boxes) > 0:
        for box in boxes:
            cv2.line(image, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
        for i,row in enumerate(detections):
            cv2.rectangle(image, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 0, 255), 2)
        imS = cv2.resize(image, (1920, 1080)) 
        cv2.imshow('img', imS)
        cv2.waitKey(0)
    '''
    return [detections, world_coords, boxes]


def calculate_angle(ymax, xmin, xmax):
    """Assumes plane on water level"""
    obj_point = ((xmax+xmin)/2, ymax)  # ymax to do bottom detection
    box_data = (np.arctan2(1520-obj_point[1], obj_point[0]-1344)*180/np.pi-90, obj_point[0], obj_point[1])
    #print(box_data)
    return box_data

def georeference(u,v, R, t_wc):
    theta = ((u - 1344) / 2688)*np.radians(109)
    psi =  ((v - 760) / 1520)*np.radians(60)
    v_c = [np.tan(theta), np.tan(psi), 1]
    v_w =  rotation_matrix(0,0,np.radians(3)) @ R@v_c + np.array([t_wc[0], t_wc[1], 0])   # 
    s = -t_wc[2] / v_w[2]
    x_w = t_wc + s*v_w
    return x_w

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
            measure = measure.T + t
            cam_measurements.append(measure)
    return cam_measurements



import numpy as np
import velodyne_decoder as vd
import matplotlib.pyplot as plt
import cv2
from PIL import Image as im
from support_func import *

def read_pcap_data(filepath):
    config = vd.Config(model='VLP-16', rpm=600)
    cloud_arrays = []
    for _, points in vd.read_pcap(filepath, config):
        cloud_arrays.append(points)

    np.save('lidar_trash_point_array.npy', np.array(cloud_arrays, dtype=object))
    return

#read_pcap_data('/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-12-48-25_Velodyne-VLP-16-Data.pcap')


# ------------------------------------------ Data cleaning ----------------------------------------------

def clean_data(lidar_data, radius, intensity, heigth, start_frame):
    for frame in lidar_data[start_frame:]:
        frame_points = []
        for point in frame:
            if np.linalg.norm([point[0], point[1]])<= radius and point[2] < heigth and point[3] > intensity: 
                frame_points.append(point)
        yield np.unique(frame_points, axis=0)


lidar_data = np.load('lidar_trash_point_array.npy', allow_pickle=True)

print('Amount of frames:', np.shape(lidar_data))
print('Data points in frame 0:', np.shape(lidar_data[0]))

lidar_generator = clean_data(lidar_data, radius = 10, intensity=1, heigth=-0.65, start_frame = 3020)
fig, ax = plt.subplots(1,2)
detector = cv2.createLineSegmentDetector(0)

for i in range(np.shape(lidar_data)[0]):

    ax[0].set_xlabel('X Label')
    ax[0].set_ylabel('Y Label')
    ax[0].set_title('Lidar data')
    ax[0].set_xlim([-10, 10])
    ax[0].set_ylim([-10, 10])
    
    ax[1].set_title('Detection zone')
    ax[1].set_xlabel('$X')
    ax[1].set_ylabel('$Y')

    thresholded_lidar_data =np.array(next(lidar_generator))
    
    print('Thresholded data points in frame ' +str(i)+  ':',np.shape(thresholded_lidar_data))
    if thresholded_lidar_data.size :

        lidar_image = lidar_to_image(thresholded_lidar_data)
        img_lines = np.zeros((100, 100), dtype=np.uint8)

        lines = detector.detect(lidar_image)
        points = clean_on_line_intersect(lines[0], thresholded_lidar_data)
        
        if type(lines[0]) == np.ndarray:
            img_lines = detector.drawSegments(img_lines, lines[0])
        
        relative_lines = get_relative_pos(lines, 'line')
        for line in relative_lines:
            ax[0].plot(line[0], line[1], color="red", linewidth=3)

        ax[1].imshow(img_lines)

        ax[0].scatter(thresholded_lidar_data[:,0], thresholded_lidar_data[:,1], 
                         marker='x', color='red')
        if np.size(points) != 0:
            ax[0].scatter(points[:,0], points[:,1], 
                         marker='o', color='blue')
    plt.pause(0.05)
    #plt.pause(0.)

    ax[0].clear()
    ax[1].clear()


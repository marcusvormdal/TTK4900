import matplotlib.pyplot as plt
import numpy as np
from support_functions.support_functions import get_relative_pos
import cv2


def initiate_plot(ax):
    ax[0][0].set_xlabel('X Label')
    ax[0][0].set_ylabel('Y Label')
    ax[0][0].set_title('Lidar data')
    ax[0][0].set_xlim([-10, 10])
    ax[0][0].set_ylim([-10, 10])
    
    ax[0][1].set_title('Line segment detector')
    ax[0][1].set_xlabel('$X')
    ax[0][1].set_ylabel('$Y')
    
    ax[1][0].set_title('Current camera frame')
    
    ax[1][1].set_xlim(-25, 25)
    ax[1][1].set_ylim(-25, 25)
    ax[1][1].set_xlabel('X')
    ax[1][1].set_ylabel('Y')
    ax[1][0].set_title('Current camera frame')
    
def full_plotter(ax, data_type, detector,  raw_lidar_data, lidar_measurements, lidar_bounds, draw_lines, 
                 camera_bounds, predictions, pos_track, camera_frame):



    if data_type == 'lidar':
        relative_plot(ax[0][0], raw_lidar_data, lidar_measurements, lidar_bounds)
        #image_plot(ax[0][1], detector, draw_lines)
        #ax[0][0].clear()
        #ax[0][1].clear()
    if data_type == 'cam':
        plot_camera_detection(ax[1][0], camera_frame, camera_bounds, predictions)
        #ax[1][0].clear()
    else:
        track_plot(ax[1][1], pos_track)
        #ax[1][1].clear()

    plt.pause(0.01)
    



def track_plot(ax, pos_track):
    x = np.array(pos_track)[:,1]
    y = np.array(pos_track)[:,2]
    ax.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=1)
         
def image_plot(ax, detector, lidar_bounds):
    if np.size(lidar_bounds)!= 0 :
        if type(lidar_bounds[0]) == np.ndarray:
            img_lines = np.zeros((100, 100), dtype=np.uint8)
            img_lines = detector.drawSegments(img_lines, lidar_bounds[0])
            ax.imshow(img_lines)

def relative_plot(ax, raw_lidar_data, lidar_measurements, line_segments):
    
    relative_lines = []
    
    if np.size(line_segments) != 0:
        relative_lines = get_relative_pos(line_segments[:,1], 'line')
    
    for line in relative_lines:
        ax.plot(line[0], line[1], color="green", linewidth=3)
        
    if np.size(raw_lidar_data) != 0:
        raw_lidar_data = np.array(raw_lidar_data)
        ax.scatter(raw_lidar_data[:,0], raw_lidar_data[:,1], 
                    marker='x', color='red', linewidths=1)
    
    if np.size(lidar_measurements) != 0:
        ax.scatter(lidar_measurements[:,0], lidar_measurements[:,1], 
                    marker='o', color='blue', linewidths=1)

def plot_camera_detection(ax, img, boxes, predictions):

    for box in boxes:
        cv2.line(img, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
        
    for row in predictions:
        cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
    
    imS = cv2.resize(img, (1920, 1080))                # Resize image
    ax.imshow(imS)

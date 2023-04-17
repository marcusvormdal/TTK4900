import matplotlib.pyplot as plt
import numpy as np
from support_functions.support_functions import get_relative_pos
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib as lib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

x, y, c = np.random.random((3, 10))
a=np.random.random((1080, 1920))
gs = gridspec.GridSpec(2, 2)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[1, :])
ln1_1, = ax1.plot([], [], color="green", linewidth=3)
ln1_2 = ax1.scatter(x, y, marker='o', color='blue', linewidths=1)
ln2, = ax2.plot([], [], color='green', marker='o', linestyle='dashed', linewidth=0.5, markersize=0.5)
ln3 = ax3.imshow(a)

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_title('Lidar data')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-10, 10])
ax2.set_title('Current camera frame')
ax2.set_xlim(-20, 20)
ax2.set_ylim(-20, 20)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax3.set_title('Current camera frame')
    
def update(frame):
    print(frame)
    data_type, raw_lidar_data, lidar_measurements, lidar_bounds, detections, pos_track, camera_frame = frame
                 
    if data_type == 'lid':
        relative_lines = []
    
        if np.size(lidar_bounds) != 0:
            
            relative_lines = get_relative_pos(lidar_bounds[:,1], 'line')
            ln1_1.set_data(relative_lines[:,0], relative_lines[:,1])
            
        #if np.size(raw_lidar_data) != 0:
        #    raw_lidar_data = np.array(raw_lidar_data)
        #    ax.scatter(raw_lidar_data[:,0], raw_lidar_data[:,1], 
        #                marker='x', color='red', linewidths=1)
        
        if np.size(lidar_measurements) != 0:
            ln1_2.set_offsets([lidar_measurements[:,1], lidar_measurements[:,0]]) #maybe
            
    if data_type == 'cam':
        for box in detections[2]:
            cv2.line(camera_frame, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
            
        for i,row in enumerate(detections[0]):
            cv2.rectangle(camera_frame, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
            cv2.putText(camera_frame, str(detections[1][i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (0, 0, 255), 2)
            
        imS = cv2.resize(camera_frame, (1920, 1080))                # Resize image
        ln3.set_array(imS)
        
    else:
        ln2.set_data(pos_track[0], pos_track[1])
        
    return ln1_1, ln1_2, ln2, ln3,
    
def plot_lsd(ax, detector, lidar_bounds):
    if np.size(lidar_bounds)!= 0 :
        if type(lidar_bounds[0]) == np.ndarray:
            img_lines = np.zeros((100, 100), dtype=np.uint8)
            img_lines = detector.drawSegments(img_lines, lidar_bounds[0])
            ax.imshow(img_lines)

def animate(animation_data):

    ani = FuncAnimation(fig, update, frames = animation_data, blit = True)
    
    plt.show()

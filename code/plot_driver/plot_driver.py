import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as lib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

x, y, c = np.random.random((3, 10))
a=np.random.random((1080, 1920))
gs = gridspec.GridSpec(2, 2)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[1, :])
ln1_1 = ax1.add_collection(LineCollection([], lw=2))
ln1_2 = ax1.scatter(x, y, marker='o', color='blue', linewidths=1)
ln2, = ax2.plot([], [], color='green', marker='o', linestyle='dashed', linewidth=0.5, markersize=0.5)
ln2_2 = ax2.scatter(100, 100, marker='x', color='red', linewidths=2)
ln2_3 = ax2.scatter(100, 100, marker='*', color='blue', linewidths=0.3)

ln3 = ax3.imshow(a)

ax1.set_xlabel('Y axis')
ax1.set_ylabel('X axis')
ax1.set_title('Lidar data')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-10, 10])
ax2.set_title('NED track')
ax2.set_xlim([-20, 20])
ax2.set_ylim([-20, 20])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax3.set_title('Current camera frame')
    
def update(frame):
    data_type, raw_lidar_data, lidar_measurements, lidar_bounds, detections, pos_track, camera_frame, ned_track = frame
                 
    if data_type == 'lid':
        if np.size(lidar_bounds) != 0:

            lidar_bounds = lidar_bounds[:,1]
            plot_l = []
            for l in lidar_bounds:
                plot_l.append([(l[0], l[1]),(l[2], l[3])])
            
            ln1_1.set_segments(plot_l)
        '''
        if np.size(raw_lidar_data) != 0:
            shape = (np.size(raw_lidar_data[:,1]),1)
        if np.size(raw_lidar_data) == 1:
            data = np.hstack((-raw_lidar_data[1], raw_lidar_data[0]))
            ln1_2.set_offsets(data)
        else:
            data = np.hstack((np.reshape(-raw_lidar_data[:,1], shape), np.reshape(raw_lidar_data[:,0], shape)))

            ln1_2.set_offsets(data)
        '''

        if np.size(lidar_measurements) != 0:
            shape = (np.size(lidar_measurements[:,1]),1)
            if np.size(lidar_measurements) == 1:
                data = np.hstack((-lidar_measurements[1], lidar_measurements[0]))
                ln1_2.set_offsets(data)
            else:
                lid_data = np.hstack((-np.reshape(lidar_measurements[:,1], shape), np.reshape(lidar_measurements[:,0], shape)))
                ln1_2.set_offsets(lid_data)

    if data_type == 'cam':
        for box in detections[2]:
            cv2.line(camera_frame, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
            
        for i,row in enumerate(detections[0]):
            cv2.rectangle(camera_frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 0, 255), 2)
            pos_str = "x : " + str(detections[1][i][0]) +" , y : " +  str(detections[1][i][1])
            cv2.putText(camera_frame, pos_str, (int(row[0]), int(row[1])-15),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        imS = cv2.resize(camera_frame, (1920, 1080))             
        ln3.set_array(camera_frame)


    else:
        pos_track = np.array(pos_track) 
        ln2.set_data(pos_track[:,0], pos_track[:,1])
        if ned_track != []:
            ned_track_lid_x = np.array([])
            ned_track_lid_y = np.array([])
            ned_track_cam_x = np.array([])
            ned_track_cam_y = np.array([])
            for meas in ned_track:
                if meas[1] == 'lid':
                    ned_track_lid_x = np.append(ned_track_lid_x, meas[0][0])
                    ned_track_lid_y = np.append(ned_track_lid_y, meas[0][1])
                elif meas[1] == 'cam':
                    ned_track_cam_x = np.append(ned_track_cam_x, meas[0][0])
                    ned_track_cam_y = np.append(ned_track_cam_y, meas[0][1])

            ned_track_lid_x = np.reshape(ned_track_lid_x, (np.size(ned_track_lid_x),1))
            ned_track_lid_y = np.reshape(ned_track_lid_y, (np.size(ned_track_lid_y),1))
            ned_track_cam_x = np.reshape(ned_track_cam_x, (np.size(ned_track_cam_x),1))
            ned_track_cam_y = np.reshape(ned_track_cam_y, (np.size(ned_track_cam_y),1))
            lid_data = np.hstack((ned_track_lid_x, ned_track_lid_y))
            cam_data = np.hstack((ned_track_cam_x, ned_track_cam_y))
            
            ln2_3.set_offsets(lid_data)
            ln2_2.set_offsets(cam_data)
            
    return ln1_1, ln1_2, ln2, ln3, ln2_2, ln2_3,
    
def plot_lsd(ax, detector, lidar_bounds):
    if np.size(lidar_bounds)!= 0 :
        if type(lidar_bounds[0]) == np.ndarray:
            img_lines = np.zeros((100, 100), dtype=np.uint8)
            img_lines = detector.drawSegments(img_lines, lidar_bounds[0])
            ax.imshow(img_lines)

def animate(animation_data):

    ani = FuncAnimation(fig, update, frames = animation_data, blit = True, interval = 100, repeat = True, save_count=2000)
    writervideo = FFMpegWriter(fps=25) 

    ani.save('C:/Users/mssvd/OneDrive/Skrivebord/TTK4900/code/animations/clustering_w_yolo7_long.mp4', writervideo)
    
    #plt.show()

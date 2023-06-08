import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as lib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.stats import chi2


x, y, c = np.random.random((3, 10))
a=np.random.random((1080, 1920))
gs = gridspec.GridSpec(6, 2)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[2:6, :]) 
ax3 = fig.add_subplot(gs[0, 1])
ln1_1 = ax1.add_collection(LineCollection([], lw=2))
ln1_2 = ax1.scatter(x, y, marker='o', color='blue', linewidths=1)
ln2, = ax2.plot([], [], color='green', marker='o', linestyle='dashed', linewidth=0.5, markersize=2.5)
ln2_2 = ax2.scatter(100, 100, marker='x', color='red', linewidths=1)
ln2_3 = ax2.scatter(100, 100, marker='+', color='blue', linewidths=1)

ln3 = ax3.imshow(a)

ax1.set_xlabel('Y axis (m)')
ax1.set_ylabel('X axis (m)')
ax1.set_title('Lidar data')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-10, 10])
ax2.set_title('NED')
ax2.set_xlim([-15, 7.5])
ax2.set_ylim([-15, 10])
ax2.legend(["USV", 'Camera', 'LiDAR'])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
#ax3.set_title('Current camera frame')
    
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
        try:
            for box in detections[2]:
                cv2.line(camera_frame, (1344, 1520), (int(box[1]), int(box[2])), (0, 255, 0), 2)
                
            for i,row in enumerate(detections[0]):
                cv2.rectangle(camera_frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 0, 255), 2)
            #    pos_str = "x : " + str(detections[1][i][0]) +" , y : " +  str(detections[1][i][1])
            #    cv2.putText(camera_frame, pos_str, (int(row[0]), int(row[1])-15),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            imS = cv2.resize(camera_frame, (1920, 1080))             
            ln3.set_array(camera_frame)
        except:
            pass

    else:
        pos_track = np.array(pos_track) 
        ln2.set_data(pos_track[:,1], pos_track[:,0])
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
            lid_data = np.hstack((ned_track_lid_y, ned_track_lid_x))
            cam_data = np.hstack((ned_track_cam_y, ned_track_cam_x))
            
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

    ani = FuncAnimation(fig, update, frames = animation_data, blit = True, interval = 100, repeat = True, save_count=5000)
    writervideo = FFMpegWriter(fps=10) 
    plt.show()
    plt.pause(0)
    name = input("run name: ")
    run_name = './animations/jpda_test_plots/'+  name  +'.mp4'
    print(run_name)
    ani.save(run_name, writervideo,  dpi=200)
    
    #plt.show()

def plot_nis(times, NIS_xy, track, confidence=0.90):
    "Modified plotting from Sensor fusion project 3 -  "
    confidence_intervals = [np.array(chi2.interval(confidence, ndof))
                            for ndof in range(1, 4)]
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.4, 5.2))
    fig.canvas.manager.set_window_title("NIS")

    ci_lower, ci_upper = confidence_intervals[2-1]
    n_total = len(NIS_xy)
    n_below = len([None for value in NIS_xy if value < ci_lower])
    n_above = len([None for value in NIS_xy if value > ci_upper])
    frac_inside = (n_total - n_below - n_above)/n_total
    frac_below = n_below/n_total
    frac_above = n_above/n_total

    ax.plot(times, NIS_xy, label=fr"$NIS_{{{'xy'}}}$")
    ax.hlines([ci_lower, ci_upper], min(times), max(times), 'C3', ":",
                    label=f"{confidence:2.1%} conf")
    print("CI --------------- ",ci_lower, ci_upper)
    ax.set_title(
        f"NIS ${{{'xy'}}}$ "
        f"({frac_inside:2.1%} inside, {frac_below:2.1%} below, "
        f"{frac_above:2.1%} above "
        f" [{confidence:2.1%} conf])")

    ax.set_yscale('log')

    ax.set_xlabel('$t$ [$s$]')
    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.1, top=0.93,
                        hspace=0.3)
    fig.savefig('NIS_track_' + str(track)+'.pdf')


def plot_nees(times, pos, track, confidence=0.90):
    "Modified plotting from Sensor fusion project 3 -  "
    ci_lower, ci_upper = np.array(chi2.interval(confidence, 2))
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.4, 9))
    fig.canvas.manager.set_window_title("NEES")

    n_total = len(pos)
    n_below = len([None for value in pos if value < ci_lower])
    n_above = len([None for value in pos if value > ci_upper])
    frac_inside = (n_total - n_below - n_above)/n_total
    frac_below = n_below/n_total
    frac_above = n_above/n_total

    ax.plot(times, pos, label=fr"$NEES_POS$")
    ax.hlines([ci_lower, ci_upper], min(times), max(times), 'C3', ":",
                    label=f"{confidence:2.1%} conf")
    ax.set_title(
        fr"NEES ${{{'POS'}}}$ "
        fr"({frac_inside:2.1%} inside, "
        f" {frac_below:2.1%} below, {frac_above:2.1%} above "
        f"[{confidence:2.1%} conf])"
    )
    ax.set_yscale('log')
    ax.set_xlabel('$t$ [$s$]')
    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.06, top=0.94,
                        hspace=0.3)
    fig.savefig('NEES_track_' + str(track)+'.pdf')
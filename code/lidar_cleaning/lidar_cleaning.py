import numpy as np
import velodyne_decoder as vd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def read_pcap_data(filepath):
    config = vd.Config(model='VLP-16', rpm=600)
    cloud_arrays = []
    for _, points in vd.read_pcap(filepath, config):
        cloud_arrays.append(points)

    np.save('lidar_trash_point_array.npy', np.array(cloud_arrays, dtype=object))
    return

#read_pcap_data('/Users/mssvd/OneDrive/Skrivebord/TTK4900/data/lidar_collection_31_01/lidar_data/2023-01-31-12-48-25_Velodyne-VLP-16-Data.pcap')


# ------------------------------------------ Data cleaning ----------------------------------------------

def clean_data(lidar_data, radius, intensity, heigth):
    for frame in lidar_data:
        frame_points = []
        for point in frame:
            if np.linalg.norm([point[0], point[1]])<= radius and point[2] < heigth and point[3] > intensity: 
                frame_points.append(point)
        yield frame_points


def hough_line(points, ):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = 20, 20
    diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
    rhos = np.linspace(int(-diag_len), int(diag_len), int(diag_len * 2))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((int(2 * diag_len), int(num_thetas)), dtype=np.uint64)

    # Vote in the hough accumulator
    for p in points:
        x = p[0]
        y = p[1]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[int(rho), int(t_idx)] += 1


    #plt.imshow(accumulator,extent=[thetas[0],thetas[-1],rhos[0],rhos[-1]],aspect='auto')

    return accumulator, thetas, rhos

lidar_data = np.load('lidar_trash_point_array.npy', allow_pickle=True)

print('Amount of frames:', np.shape(lidar_data))
print('Data points in frame 0:', np.shape(lidar_data[0]))


lidar_generator = clean_data(lidar_data, radius = 10, intensity=1, heigth=-0.65)
fig, ax = plt.subplots(1,2)

for i in range(np.shape(lidar_data)[0]):
    ax[0].set_xlabel('X Label')
    ax[0].set_ylabel('Y Label')
    ax[0].set_title('Lidar data')
    ax[0].set_xlim([-10, 10])
    ax[0].set_ylim([-10, 10])
    
    ax[1].set_title('Accumulatorarray')
    ax[1].set_xlabel('$\\theta$(radians)')
    ax[1].set_ylabel('$\\rho$(pixels)')

    
    thresholded_lidar_data = np.array(next(lidar_generator))
    print(thresholded_lidar_data)
    print('Thresholded data points in frame ' +str(i)+  ':',np.shape(thresholded_lidar_data))
    if thresholded_lidar_data != []:
        ax[0].scatter(thresholded_lidar_data[:,0], thresholded_lidar_data[:,1], marker='o', color='blue')
        
        accumulator, thetas, rhos = hough_line(thresholded_lidar_data)

        ax[1].imshow(accumulator, extent =[thetas[0], thetas[-1], rhos[0], rhos[-1]])

    plt.pause(0.1)
    fig.clf()

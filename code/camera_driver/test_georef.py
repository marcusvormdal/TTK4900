
import numpy as np
import cv2
from cv_bridge import CvBridge
import glob
import rosbag
import gpxpy
import gpxpy.gpx
from datetime import datetime, timedelta
bag = rosbag.Bag('../bags/georef_test2.bag', "r")

bridge = CvBridge()

#distorted
#box 1 (0,(1684+1706)/2,923)
#box 2 (0,(765+794)/2,957)
#box 3 (0,(87+144)/2,1148)
#box 4 (1267+1373)/2,1520)
#box 5 (1758+1796)/2,1042)
#box 6(1184+1207)/2,923)

#undistorted

#box 1 (0,(1690+1712)/2,924)      
#box 2 (0,(744+777)/2,961)         
#box 3                             
#box 4 (0,(1262+1375)/2,1520)      
#box 5 (0,(1769+918)/2,1048)        
#box 6 (0,(1183+1208)/2,922)        


#fish

(0, (1560+1454)/2, 873)

(0,  (935+912)/2, 900) 

(0, (255+181)/2, 1105) 

(0, (1335+1257)/2, 1315) 

(0, (1629+1600)/2, 959)

(0, (1225+1208)/2, 871) 


#fish 2
# mostright hhm3 (0, 1630, 916)  
# lef 

def test_world_coord():
    psi,theta,phi = np.radians(-90),np.radians(0),np.radians(90)
    box_coord  = (0,(1769+1810)/2,1048)                        
    
    R = rotation_matrix(psi,theta,phi)
    
    world_cord = alternate_world_coord(box_coord[1],box_coord[2], R, np.array([0.0,0.0,0.705]))
    print(world_cord)

def alternate_world_coord(u,v, R, t_wc):
    theta = ((u - 1344) / 2688)*np.radians(109)
    psi =  ((v - 760) / 1520)*np.radians(60)
    v_c = [np.tan(theta), np.tan(psi), 1]
    v_w = rotation_matrix(0,0,np.radians(1)) @ R@v_c + np.array([t_wc[0], t_wc[1], 0])
    s = -t_wc[2] / v_w[2]
    x_w = t_wc + s*v_w
    
    x_w[0] = x_w[0]+0.24
    print("x_w",x_w)
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

def gt_generator():
    gpx_file = open('../../data/test_marcus/gnns_data/20230516-132518-loc3.gpx', 'r')
    gpx = gpxpy.parse(gpx_file)
    lat = []
    long = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                print("time", datetime.timestamp(point.time))
                lat.append(point.latitude)
                long.append(point.longitude)
                
    print("lat", np.mean(np.array(lat)))
    print("long",np.mean(np.array(long)))
    
intrinsic = cv2.UMat(np.array([[2.75344274e+03, 0.00000000e+00, 1.34664016e+03],[0.00000000e+00, 2.77845260e+03, 7.48362894e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])) 

distortion = cv2.UMat(np.array([[-0.94911185,  2.27298045 , 0.02827832 ,-0.00913316 ,-3.70567064]]))


def undistort(cv_img, balance=1.0, dim2=None, dim3=None):
    K=np.array([[1673.1769976174614, 0.0, 1293.237778062828], [0.0, 1686.9412681651288, 768.7938516098573], [0.0, 0.0, 1.0]])
    D=np.array([[-0.17432046444438237], [0.1907989411768662], [-0.28952863222406533], [0.20427818635225561]])
    img = cv_img
    K=np.array(K)
    D=np.array(D)
    DIM=img.shape[:2][::-1]
    print(DIM)
    dim1 = DIM  #dim1 is the dimension of input image to un-distort
    dim2 = dim1
    dim3 = dim1
    scaled_K = K   # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def __main__():
    
    test_world_coord()
    #gt_generator()
    
    #for topic, msg, t in bag.read_messages(topics=['/camera/image_raw/compressed']):
        
    #    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #cv_img = cv2.undistort(cv_img, intrinsic, distortion, None, None)
        #cv_img = cv2.UMat.get(cv_img)
    #    cv2.imshow('img', cv_img)
    #   undistort(cv_img)
        
    '''
    image_res = cv2.resize(cv_img, (672, 380))
    cv2.imshow('image',image_res)
    cv2.waitKey(0)
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (2688,1520), 1, (2688,1520))
    print(newcameramtx)
    print(roi)
    
    
    cv_img = cv2.undistort(cv_img, intrinsic, distortion, None, newcameramtx)
    cv_img = cv2.UMat.get(cv_img)
    cv2.imshow('img', cv_img)
    cv2.waitKey(0)
    
    
    #cv_img = cv2.undistort(cv_img, intrinsic, distortion, None)
    #image_res = cv2.resize(cv_img, (672, 380))

    #cv2.imshow('img', image_res)
    #cv2.waitKey(0)
    '''
        
__main__()






import numpy as np
import cv2
import glob

def run_img_calibration():
    '''from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    objpoints = [] 
    imgpoints = [] 
    images = glob.glob('*.png')
    for fname in images:
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (8,6), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx)
    print(dist)

    img = cv2.imread('./035.png')
    
    dst = cv2.undistort(img, mtx, dist, None)
    
    cv2.imshow('img', dst)
    cv2.imwrite('./results/overleaf35.png', dst)
    cv2.waitKey(0)
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    
    
intrinsic = cv2.UMat(np.array([[2.75344274e+03, 0.00000000e+00, 1.34664016e+03],[0.00000000e+00, 2.77845260e+03, 7.48362894e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])) 

distortion = cv2.UMat(np.array([[-0.94911185,  2.27298045 , 0.02827832 ,-0.00913316 ,-3.70567064]]))
    
def __main__():
    run_img_calibration()
    
    
__main__()





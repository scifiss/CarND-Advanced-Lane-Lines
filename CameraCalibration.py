# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:01:33 2017

@author: gaor
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from scipy import misc
import pickle
#----------------------------------------
## camera calibration
#----------------------------------------
image = mpimg.imread('camera_cal\calibration1.jpg')
height, width, chann = image.shape
#plt.imshow(image)

# chess pattern
nx=9
ny=6
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# same for all calibration images
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calImages = glob.glob('./camera_cal/calibration*.jpg')
for idx, filename in enumerate(calImages):   
    image = mpimg.imread( filename)
    imagename = filename.split('\\')[-1]
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    if ret == True:
        image_corners = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        #plt.imshow(image_corners)
        cv2.imwrite('./camera_cal/FoundCorners_'+str(idx)+'_'+imagename,image_corners)
        # save one resulted image to output        
        #misc.imsave('output_images/' + 'image_withDetectedCorners.jpg', image_corners)
        # 'cornerSubPix' is supposed to refine the corner locations, but I see no difference in corners and corners2 for each image
        #corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #image_corners = cv2.drawChessboardCorners(image, (nx,ny), corners2, ret)
        imgpoints.append(corners)
        objpoints.append(objp)
        #cv2.waitKey(500)
        
#cv2.destroyAllWindows()
        
# calibration with the following outputs:
# mtx: camera matrix
# dist: distortion coeff
# rvecs and tvecs: rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

# save the results for later use by serializing the object hierarchy using pickle
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist']= dist
pickle.dump( dist_pickle, open('./calibration_pickle.p','wb'))
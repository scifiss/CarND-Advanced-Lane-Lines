# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:06:01 2017

@author: Rebecca
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
# CameraCalibration.py



camCali_data = pickle.load(open('./calibration_pickle.p','rb'))
dist = camCali_data['dist']
mtx = camCali_data['mtx']

image = mpimg.imread('camera_cal\calibration1.jpg')
height, width, chann = image.shape


def gen_mask(img, threshold):
    themask = np.zeros_like(img)
    themask[(img>=threshold[0]) & (img<=threshold[1])] = 1
    return themask

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = gen_mask(absgraddir, thresh)

    return binary_output


def color_threshold(img, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #s
    s_binary = gen_mask(hls[:,:,2], sthresh)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_binary = gen_mask(hsv[:,:,2], vthresh)
    
    output = np.zeros_like(s_binary)
    output[(s_binary==1) & (v_binary==1)] = 1    
    
    return output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F,1,0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F,0,1))
    # Convert the absolute value image to 8-bit:
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = gen_mask(scaled_sobel, thresh)
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255    
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = gen_mask(gradmag, thresh)
    
    return binary_output

testImages = glob.glob('./test_images/test*.jpg')

for idx, filename in enumerate(testImages):
    img = mpimg.imread( filename)  # read as RGB
    imagename = filename.split('\\')[-1]

#----------------------------------------
## distortion correction
#----------------------------------------    
    undist = cv2.undistort(img, mtx, dist, None, mtx) # show as RGB  
    
#----------------------------------------
## gradient & color threshold
#----------------------------------------        
    # generate masked images that highlight the lanes
    preprocessed = np.zeros_like(img[:,:,0])
    # gradient on x and y direction
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 255))    
   
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 255))   
    
    # magnitude of the gradient at 45degree
    mag_binary = mag_thresh(img, sobel_kernel=3, thresh=(30, 100))
   
    # direction of the gradient
    dir_binary = dir_threshold(img, sobel_kernel = 15, thresh=(0.5,1))
    
    # color threshold on saturation and value channel
    col_binary = color_threshold(img, sthresh=(100,255),vthresh=(50,255))
    masked = ((gradx==1) & (grady==1)) | (col_binary==1)
    preprocessed[masked] = 255
    mpimg.imsave('./test_images/preprocessed_'+imagename,preprocessed,cmap='gray')  
    
    btm_width =  0.57  #0.76
    top_width = 0.08 
    top_pct = 0.63  # 0.62
    btm_pct = 0.914 # 0.935
    src = np.float32([ [width * (0.5-top_width/2), height*top_pct], [width*(0.5+top_width/2), height*top_pct], \
                      [width * (0.5-btm_width/2), height*btm_pct], [width*(0.5+btm_width/2), height*btm_pct] ])
    offset = width*0.33  #width* (0.5-btm_width/2)
    dst =np.float32([ [offset,0], [width-offset,0], [width-offset, height], [offset, height]])
    
#----------------------------------------
## perspective transform
#----------------------------------------
    M = cv2.getPerspectiveTransform(src, dst) 
    warped = cv2.warpPerspective(preprocessed, M, ( width, height), flags=cv2.INTER_LINEAR)
    mpimg.imsave('./test_images/warped_'+imagename,warped,cmap='gray')  

    
#----------------------------------------
## detect lane lines

## determine lane curvature

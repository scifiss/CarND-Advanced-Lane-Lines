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


## Define a function that takes an image, number of x and y points, 
## camera matrix and distortion coefficients
#def corners_unwarp(img, nx, ny, mtx, dist):
#    # Use the OpenCV undistort() function to remove distortion
#    undist = cv2.undistort(img, mtx, dist, None, mtx)
#    # Convert undistorted image to grayscale
#    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
#    # Search for corners in the grayscaled image
#    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#    if ret == True:
#        # If we found corners, draw them! (just for fun)
#        image_corners=cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
#        misc.imsave('output_images/' + 'undistortedimage_withDetectedCorners.jpg', image_corners)
#        # Choose offset from image corners to plot detected corners
#        # This should be chosen to present the result at the proper aspect ratio
#        # My choice of 100 pixels is not exact, but close enough for our purpose here
#        offset = 100 # offset for dst points
#        # Grab the image shape
#        img_size = (gray.shape[1], gray.shape[0])
#
#        # For source points I'm grabbing the outer four detected corners
#        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#        # For destination points, I'm arbitrarily choosing some points to be
#        # a nice fit for displaying our warped result 
#        # again, not exact, but close enough for our purposes
#        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
#                                     [img_size[0]-offset, img_size[1]-offset], 
#                                     [offset, img_size[1]-offset]])
#        # Given src and dst points, calculate the perspective transform matrix
#        M = cv2.getPerspectiveTransform(src, dst)
#        # Warp the image using OpenCV warpPerspective()
#        warped = cv2.warpPerspective(undist, M, img_size)
#
#    # Return the resulting image and matrix
#    return warped, M

#img = mpimg.imread('camera_cal\calibration3.jpg')
#top_down, perspective_M = corners_unwarp(image, nx, ny, mtx, dist)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(top_down)
#ax2.set_title('Undistorted and Warped Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


testImages = glob.glob('./test_images/test*.jpg')


for idx, filename in enumerate(testImages):
    img = mpimg.imread( filename)  # read as RGB
    imagename = filename.split('\\')[-1]

#----------------------------------------
## distortion correction
#----------------------------------------    
    undist = cv2.undistort(img, mtx, dist, None, mtx) # show as RGB
    mpimg.imsave('./test_images/Undistorted_'+imagename,undist)
    
#----------------------------------------
## gradient & color threshold
#----------------------------------------        
    # generate masked images that highlight the lanes
    preprocessed = np.zeros_like(img[:,:,0])
    # gradient on x and y direction
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 255))    
    mpimg.imsave('./test_images/gradx_'+imagename,gradx,cmap='gray')  
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 255))   
    mpimg.imsave('./test_images/grady_'+imagename,grady,cmap='gray')  
    # magnitude of the gradient at 45degree
    mag_binary = mag_thresh(img, sobel_kernel=3, thresh=(30, 100))
    mpimg.imsave('./test_images/mag_'+imagename,mag_binary,cmap='gray')  
    # direction of the gradient
    dir_binary = dir_threshold(img, sobel_kernel = 15, thresh=(0.5,1))
    mpimg.imsave('./test_images/dir_'+imagename,dir_binary,cmap='gray')  

    # color threshold on saturation and value channel
    col_binary = color_threshold(img, sthresh=(100,255),vthresh=(50,255))
    masked = ((gradx==1) & (grady==1)) | (col_binary==1)
    preprocessed[masked] = 255
    mpimg.imsave('./test_images/preprocessed_'+imagename,preprocessed,cmap='gray')  
    
    btm_width = 0.76 # 0.57  #0.76
    top_width = 0.08 
    top_pct = 0.62 #0.63  # 0.62
    btm_pct = 0.935  #0.914 # 0.935
    src = np.float32([ [width * (0.5-top_width/2), height*top_pct], [width*(0.5+top_width/2), height*top_pct], \
                      [width * (0.5-btm_width/2), height*btm_pct], [width*(0.5+btm_width/2), height*btm_pct] ])
    offset = width*0.33  #width* (0.5-btm_width/2)
    dst =np.float32([ [offset,0], [width-offset,0], [width-offset, height], [offset, height]])
    
#----------------------------------------
## perspective transform
#----------------------------------------
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessed, M, ( width, height), flags=cv2.INTER_LINEAR)
    mpimg.imsave('./test_images/warped_'+imagename,warped,cmap='gray')  





    


    
#----------------------------------------
## detect lane lines
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


## determine lane curvature

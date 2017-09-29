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

  
#----------------------------------------
## gradient & color threshold
#----------------------------------------        

#----------------------------------------
## perspective transform
#----------------------------------------
 
# UndistAndPerspTransf.py
def gen_mask(img, threshold):
    themask = np.zeros_like(img)
    themask[(img>=threshold[0]) & (img<=threshold[1])] = 1
    return themask
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

def transformPipeline(img, mtx, dist, sthresh, vthresh,sobel_kernel, xthresh, ythresh, perspective_M):
    
    height, width, chann = img.shape

    undist = cv2.undistort(img, mtx, dist, None, mtx) # show as RGB
    preprocessed = np.zeros_like(undist[:,:,0])
    # gradient on x and y direction
    gradx = abs_sobel_thresh(undist, 'x', sobel_kernel, xthresh)    
    grady = abs_sobel_thresh(undist, 'y', sobel_kernel, ythresh) 
    col_binary = color_threshold(undist, sthresh,vthresh)
    masked = ((gradx==1) & (grady==1)) | (col_binary==1)
    preprocessed[masked] = 1
    
    unwarped = cv2.warpPerspective(preprocessed, perspective_M, ( width, height), flags=cv2.INTER_LINEAR)
    return unwarped

    
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
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)



## determine lane curvature

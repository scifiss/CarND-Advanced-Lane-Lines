# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:06:01 2017

@author: Rebecca
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import misc
import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#----------------------------------------
## camera calibration
#----------------------------------------
# CameraCalibration.py
camCali_data = pickle.load(open('./calibration_pickle.p','rb'))
dist = camCali_data['dist']
mtx = camCali_data['mtx']

Warp_data=pickle.load(open('./Warp_pickle.p','rb'))
M = Warp_data['M']
Minv = Warp_data['Minv']  
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
nlastfit = 3
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
    def update_fit(self, newfit, inds):
        # add a new fit to the line fitting history, up to nlastfit recent fits
        if newfit is not None:
            if self.best_fit is not None:                
                self.diffs = abs(newfit-self.best_fit)
            if self.diffs[0] > 0.0001 and len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                #self.allx = np.count_nonzero(inds)
                self.current_fit.append(newfit)
                if len(self.current_fit) > nlastfit:
                    # throw out old fits, keep newest 
                    self.current_fit = self.current_fit[len(self.current_fit)-nlastfit:]
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
                
def findingLanesbyHist(binary_warped):
    height, width = binary_warped.shape
    # the original 0-1 image is read as 0-255
    if np.max(binary_warped==255):
        binary_warped = binary_warped//255
    #plt.imshow(binary_warped,cmap='gray')
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #mpimg.imsave('./test_images/stacked_test1.jpg',out_img)  
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting

    return left_fit, right_fit,left_lane_inds,right_lane_inds

def findingLanesbyPrefit(binary_warped,left_fit,right_fit):
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 70
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                     & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                     & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)

    return left_fit_new, right_fit_new,left_lane_inds,right_lane_inds

def CalCurvatureAndDistance(left_fit, right_fit,ploty,left_fitx,right_fitx, width, height):
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    avg_curverad = (left_curverad+right_curverad)/2
    
    carPos = width/2
    left_x = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_x = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    laneCenter = (left_x + right_x) /2
    DistanceInLife = np.abs(carPos-laneCenter)*xm_per_pix    
    LeftOfLaneCenter = carPos<laneCenter
    return avg_curverad, DistanceInLife, LeftOfLaneCenter

def VisualizeLane(undist,ploty,left_fitx,right_fitx,width, height,avg_curverad,DistanceInLife,LeftOfLaneCenter):
    result = np.copy(undist)
    warp_zero = np.zeros((height, width)).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#    undist = cv2.imread('./test_images/undistorted_test2.jpg')
#    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
    # Combine the result with the original image
    result = cv2.addWeighted(result, 1, newwarp, 0.3, 0)
    
    info = 'Radius of Curvature = '+ '{:05.3f}'.format(avg_curverad)+'m'
    cv2.putText(result, info, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3, 8)
    pos = 'right'
    if LeftOfLaneCenter:
        pos = 'left'
    info = 'Vehicle is '+'{:04.2f}'.format(DistanceInLife)+' '+pos+' of center'
    cv2.putText(result, info, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3, 8)
    return result

    
def process_image(image):
    height, width, chann = image.shape

    undist = cv2.undistort(image, mtx, dist, None, mtx) # show as RGB
    preprocessed = np.zeros_like(undist[:,:,0])
    # gradient on x and y direction
    gradx = abs_sobel_thresh(undist, 'x', sobel_kernel=3, thresh=(12, 255))    
    grady = abs_sobel_thresh(undist, 'y', sobel_kernel=3, thresh=(25, 255)) 
    col_binary = color_threshold(undist, sthresh=(100,255),vthresh=(50,255))
    masked = ((gradx==1) & (grady==1)) | (col_binary==1)
    preprocessed[masked] = 1
    
    unwarped = cv2.warpPerspective(preprocessed, M, ( width, height), flags=cv2.INTER_LINEAR)
    
    if not leftLine.detected or not rightLine.detected:
        left_fit, right_fit,left_lane_inds,right_lane_inds = findingLanesbyHist(unwarped)
        
    else:
        left_fit, right_fit,left_lane_inds,right_lane_inds = findingLanesbyPrefit(unwarped, leftLine.best_fit, rightLine.best_fit)
        # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
#    if left_fit is not None and left_fit is not None:
#        # calculate x-intercept (bottom of image, x=image_height) for fits
#        
#        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
#        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
#        x_int_diff = abs(right_fit_x_int-left_fit_x_int)
#        if abs(350 - x_int_diff) > 100:
#            left_fit = None
#            right_fit = None
            
    leftLine.update_fit(left_fit, left_lane_inds)
    rightLine.update_fit(right_fit, right_lane_inds)
    
    # draw the current best fit if it exists
    if leftLine.best_fit is not None and rightLine.best_fit is not None:
        ploty = np.linspace(0, height-1, height )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        avg_curverad, DistanceInLife, LeftOfLaneCenter = CalCurvatureAndDistance(left_fit, right_fit,ploty,left_fitx,right_fitx, width, height)
        result = VisualizeLane(undist,ploty,left_fitx,right_fitx,width, height,avg_curverad,DistanceInLife,LeftOfLaneCenter)
        
        
    else:
        result = np.copy(image)
    
#    diagnostic_output = False
#    if diagnostic_output:
#        # put together multi-view output
#        diag_img = np.zeros((720,1280,3), dtype=np.uint8)
#        
#        # original output (top left)
#        diag_img[0:360,0:640,:] = cv2.resize(img_out,(640,360))
#        
#        # binary overhead view (top right)
#        img_bin = np.dstack((img_bin*255, img_bin*255, img_bin*255))
#        resized_img_bin = cv2.resize(img_bin,(640,360))
#        diag_img[0:360,640:1280, :] = resized_img_bin
#        
#        # overhead with all fits added (bottom right)
#        img_bin_fit = np.copy(img_bin)
#        for i, fit in enumerate(leftLine.current_fit):
#            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))
#        for i, fit in enumerate(rightLine.current_fit):
#            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0,20*i+100,20*i+100))
#        img_bin_fit = plot_fit_onto_img(img_bin_fit, leftLine.best_fit, (255,255,0))
#        img_bin_fit = plot_fit_onto_img(img_bin_fit, rightLine.best_fit, (255,255,0))
#        diag_img[360:720,640:1280,:] = cv2.resize(img_bin_fit,(640,360))
#        
#        # diagnostic data (bottom left)
#        color_ok = (200,255,155)
#        color_bad = (255,155,155)
#        font = cv2.FONT_HERSHEY_DUPLEX
#        if left_fit is not None:
#            text = 'This fit L: ' + ' {:0.6f}'.format(left_fit[0]) + \
#                                    ' {:0.6f}'.format(left_fit[1]) + \
#                                    ' {:0.6f}'.format(left_fit[2])
#        else:
#            text = 'This fit L: None'
#        cv2.putText(diag_img, text, (40,380), font, .5, color_ok, 1, cv2.LINE_AA)
#        if right_fit is not None:
#            text = 'This fit R: ' + ' {:0.6f}'.format(right_fit[0]) + \
#                                    ' {:0.6f}'.format(right_fit[1]) + \
#                                    ' {:0.6f}'.format(right_fit[2])
#        else:
#            text = 'This fit R: None'
#        cv2.putText(diag_img, text, (40,400), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Best fit L: ' + ' {:0.6f}'.format(leftLine.best_fit[0]) + \
#                                ' {:0.6f}'.format(leftLine.best_fit[1]) + \
#                                ' {:0.6f}'.format(leftLine.best_fit[2])
#        cv2.putText(diag_img, text, (40,440), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Best fit R: ' + ' {:0.6f}'.format(rightLine.best_fit[0]) + \
#                                ' {:0.6f}'.format(rightLine.best_fit[1]) + \
#                                ' {:0.6f}'.format(rightLine.best_fit[2])
#        cv2.putText(diag_img, text, (40,460), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Diffs L: ' + ' {:0.6f}'.format(leftLine.diffs[0]) + \
#                             ' {:0.6f}'.format(leftLine.diffs[1]) + \
#                             ' {:0.6f}'.format(leftLine.diffs[2])
#        if leftLine.diffs[0] > 0.001 or \
#           leftLine.diffs[1] > 1.0 or \
#           leftLine.diffs[2] > 100.:
#            diffs_color = color_bad
#        else:
#            diffs_color = color_ok
#        cv2.putText(diag_img, text, (40,500), font, .5, diffs_color, 1, cv2.LINE_AA)
#        text = 'Diffs R: ' + ' {:0.6f}'.format(rightLine.diffs[0]) + \
#                             ' {:0.6f}'.format(rightLine.diffs[1]) + \
#                             ' {:0.6f}'.format(rightLine.diffs[2])
#        if rightLine.diffs[0] > 0.001 or \
#           rightLine.diffs[1] > 1.0 or \
#           rightLine.diffs[2] > 100.:
#            diffs_color = color_bad
#        else:
#            diffs_color = color_ok
#        cv2.putText(diag_img, text, (40,520), font, .5, diffs_color, 1, cv2.LINE_AA)
#        text = 'Good fit count L:' + str(len(leftLine.current_fit))
#        cv2.putText(diag_img, text, (40,560), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Good fit count R:' + str(len(rightLine.current_fit))
#        cv2.putText(diag_img, text, (40,580), font, .5, color_ok, 1, cv2.LINE_AA)
#        
#        img_out = diag_img
        return result


    #if not leftLine.    
## determine lane curvature

leftLine = Line()
rightLine = Line()
#my_clip.write_gif('test.gif', fps=12)

video_input = VideoFileClip('project_video.mp4')#.subclip(22,26)
video_clip = video_input.fl_image(process_image)
video_clip.write_videofile('project_video_output.mp4',audio=False)
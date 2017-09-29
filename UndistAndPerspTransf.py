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

def hls_threshold(img, hthresh=(0,180), lthresh=(0,255), sthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_binary = gen_mask(hls[:,:,2], sthresh)
    h_binary = gen_mask(hls[:,:,0], hthresh)
    l_binary = gen_mask(hls[:,:,1], lthresh)
    output = np.zeros_like(s_binary)
    output[(s_binary==1) & (l_binary==1) & (h_binary==1)] = 1    
    return output

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
    mpimg.imsave('./test_images/Undistorted_'+imagename,undist)
    
#----------------------------------------
## gradient & color threshold
#----------------------------------------        
    # generate masked images that highlight the lanes
    preprocessed = np.zeros_like(img[:,:,0])
    
    hlsthreshed = hls_threshold(undist, hthresh=(0,180), lthresh=(0,255), sthresh=(150,255))
    mpimg.imsave('./test_images/hls_s_'+imagename,hlsthreshed,cmap='gray')  
    hlsthreshed = hls_threshold(undist, hthresh=(0,80), lthresh=(0,255), sthresh=(0,255))
    mpimg.imsave('./test_images/hls_h_'+imagename,hlsthreshed,cmap='gray')  
    hlsthreshed = hls_threshold(undist, hthresh=(0,180), lthresh=(100,255), sthresh=(0,255))
    mpimg.imsave('./test_images/hls_l_'+imagename,hlsthreshed,cmap='gray')  
    
    # gradient on x and y direction
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=3, thresh=(12, 255))    
    mpimg.imsave('./test_images/gradx_'+imagename,gradx,cmap='gray')  
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=3, thresh=(25, 255))   
    mpimg.imsave('./test_images/grady_'+imagename,grady,cmap='gray')  
    # magnitude of the gradient at 45degree
    mag_binary = mag_thresh(undist, sobel_kernel=3, thresh=(30, 100))
    mpimg.imsave('./test_images/mag_'+imagename,mag_binary,cmap='gray')  
    # direction of the gradient
    dir_binary = dir_threshold(undist, sobel_kernel = 15, thresh=(0.5,1.0))
    mpimg.imsave('./test_images/dir_'+imagename,dir_binary,cmap='gray')  
    # color threshold on saturation and value channel
    col_binary = color_threshold(undist, sthresh=(100,255),vthresh=(50,255))
    mpimg.imsave('./test_images/col_'+imagename,col_binary,cmap='gray')
    
    # combined thresholding
    masked = ((gradx==1) & (grady==1)) | (col_binary==1)
    preprocessed[masked] = 1
    mpimg.imsave('./test_images/preprocessed_'+imagename,preprocessed,cmap='gray')  
    
#    btm_width =  0.7063  #0.57  #0.76
#    top_width = 0.0445  #0.08 
#    top_pct = 0.61  # 0.62
#    btm_pct = 1  #0.914 # 0.935
    bottom_left = [320,720] 
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]
#    src = np.float32([ [width * (0.5-top_width/2), height*top_pct], [width*(0.5+top_width/2), height*top_pct], \
#                      [width * (0.5+btm_width/2), height*btm_pct], [width*(0.5-btm_width/2), height*btm_pct] ])
    #src = np.float32([[630,439],  [687,439], [1155,720], [251,720]])
    src = np.float32([[570,468],  [714,468], [1106,720], [207,720]])
    #offset = width*0.33  #width* (0.5-btm_width/2)
    #dst =np.float32([ [offset,1], [width-offset,1], [width-offset, height], [offset, height]])
    dst = np.float32([top_left,top_right,bottom_right, bottom_left])   
#----------------------------------------
## perspective transform
#----------------------------------------
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(preprocessed, M, ( width, height), flags=cv2.INTER_LINEAR)
    mpimg.imsave('./test_images/warped_'+imagename,unwarped,cmap='gray')  

Warp_pickle = {}
Warp_pickle['M'] = M
Warp_pickle['Minv']= Minv
pickle.dump( Warp_pickle, open('./Warp_pickle.p','wb'))
# show the result
#warped0 = cv2.warpPerspective(image, M, ( width, height), flags=cv2.INTER_LINEAR)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(warped0)
#ax2.set_title('Undistorted and Warped Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.savefig('output_images/warped_straight_line1.jpg')
#
#perspective_M = M
fig, axs = plt.subplots(2,2,)
fig.subplots_adjust(hspace=0.01,wspace=0.01)
i=0
axs[i,0].imshow(mpimg.imread( './test_images/test2.jpg') )
axs[i,0].set_title('test2')
axs[i,0].axis('off')
axs[i,1].imshow(mpimg.imread( './test_images/test5.jpg') )
axs[i,1].axis('off')
axs[i,1].set_title('test5')
i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/Undistorted_test2.jpg') )
axs[i,0].set_title('undistorted')
axs[i,0].axis('off')
axs[i,1].imshow(mpimg.imread( './test_images/Undistorted_test5.jpg') )
axs[i,1].set_title('undistorted')
axs[i,1].axis('off')
plt.savefig('output_images/two_test_images.jpg', bbox_inches='tight')
    
fig, axs = plt.subplots(7,2,figsize=(30,60))
fig.subplots_adjust(hspace=0.1,wspace=0.001)
img1 = mpimg.imread( './test_images/Undistorted_test2.jpg') 
i=0

axs[i,0].imshow(mpimg.imread( './test_images/hls_l_test2.jpg') )
axs[i,0].set_title('HLS L channel')
axs[i,1].imshow(mpimg.imread( './test_images/hls_l_test5.jpg') )
axs[i,1].set_title('HLS L channel')
i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/hls_s_test2.jpg') )
axs[i,0].set_title('HLS S channel')
axs[i,1].imshow(mpimg.imread( './test_images/hls_s_test5.jpg') )
axs[i,1].set_title('HLS S channel')
i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/dir_test2.jpg') )
axs[i,0].set_title('Dir channel')
axs[i,1].imshow(mpimg.imread( './test_images/dir_test5.jpg') )
axs[i,1].set_title('Dir channel')
i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/mag_test2.jpg') )
axs[i,0].set_title('Mag channel')
axs[i,1].imshow(mpimg.imread( './test_images/mag_test5.jpg') )
axs[i,1].set_title('Mag channel')

i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/gradx_test2.jpg') )
axs[i,0].set_title('Gradx with Sobel')
axs[i,1].imshow(mpimg.imread( './test_images/gradx_test5.jpg') )
axs[i,1].set_title('Gradx with Sobel')

i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/grady_test2.jpg') )
axs[i,0].set_title('Grady with Sobel')
axs[i,1].imshow(mpimg.imread( './test_images/grady_test5.jpg') )
axs[i,1].set_title('Grady with Sobel')

i=i+1
axs[i,0].imshow(mpimg.imread( './test_images/preprocessed_test2.jpg') )
axs[i,0].set_title('Preprocessed')
axs[i,1].imshow(mpimg.imread( './test_images/preprocessed_test5.jpg') )
axs[i,1].set_title('Preprocessed')
plt.savefig('output_images/Preprocessing.jpg', bbox_inches='tight')


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(mpimg.imread( './test_images/warped_test2.jpg') )
ax1.set_title('Unwarped test2', fontsize=50)
ax2.imshow(mpimg.imread( './test_images/warped_test5.jpg') )
ax2.set_title('Unwarped test5', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('output_images/Unwarped_images.jpg')
#----------------------------------------


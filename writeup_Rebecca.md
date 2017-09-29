**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/FoundCorners_calibration3.jpg "corners"
[image1]: ./output_images/Undist_calibration3.jpg "Undistorted"
[image2]: ./output_images/two_test_images.jpg "Test images undistorted"
[image3]: ./output_images/Preprocessing.jpg "Binary Example"
[image4]: ./output_images/Undistorted_straight_lines1_marked.jpg "Work out src points"
[image5]: ./output_images/Unwarped_images.jpg "Unwarp Example"
[image6]: ./output_images/warped_straight_line1.jpg "Unwarp the straight line"
[image7]: ./output_images/hist_warped_test1.png "histogram"
[image8]: ./output_images/findLanesNCurves.jpg "Fit curve"
[image9]: ./output_images/findCurvesNVisualize.jpg "Curve visualized"
[image10]: ./output_images/LaneVisualized.jpg "Lane visualized"
[video1]: ./project_video_output.mp4 "Project Video"
[image11]: ./output_images/challenge_video_image.png "challenge video"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `CameraCalibration.py` in lines 22 through 56.  

I start by preparing object points `objpoints`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I also tried `cornerSubPix()` to improve the accuracy of the corner, but it turned out for every image, there is no difference between `corners` and `corners2`
```
for idx, filename in enumerate(calImages):   
    image = mpimg.imread( filename)
    imagename = filename.split('\\')[-1]
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    if ret == True:
        image_corners = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        # 'cornerSubPix' is supposed to refine the corner locations, but I see no difference in corners and corners2 for each image
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #image_corners = cv2.drawChessboardCorners(image, (nx,ny), corners2, ret)
        imgpoints.append(corners)
        objpoints.append(objp)
```
One image detected with corners is shown below.

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function with camera matrix `mtx` and distortion coefficients `dist` and obtained this result: 
```
undist = cv2.undistort(img, mtx, dist, None, mtx)
```
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Here are examples of undistorted 'test2' and 'test5' images for the following steps.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Many attempts are made to threshold the undistorted images to be prepared for unwarping. 
- The RBG image is converted to HLS and HSV colorspace. The lightness `L` channel or value `V` channel works fine even when shadows occur, since the lane line is always white/yellow, but it fails when the color of the road is bright with light colors, like test 5. The saturation `S` channel, which means how colorful, works much better with the high saturated lines. 
- The union combination of gradients at x and y axis direction works good, which uses `Sobel` kernel on the gray image.
- The direction thresholded image works based on `Sobel` operator, and the range for the radian of lane's angle is between 0.5 and 1. It turns out fails in most cases.
- The magnitude of the gradient at 45 degree direction also does a good job in identifying lane-lines, but it is sensitive to textures on the road.

Finally I used a combination of S channel, V channel, and gradient thresholds to generate a binary image (thresholding steps at lines 29 through 149 in `UndistAndPerspTransf.py`).  
Here's an example of my output for this step on 'test2' and 'test5' images.  
* 1st row: HLS L channel thresholded result
* 2nd row: HLS S channel thresholded result
* 3rd row: thresholded on direction of gradient
* 4th row: thresholded on magnitude of gradient
* 5th row: thresholded on x-direction of gradient
* 6th row: thresholded on y-direction of gradient
* 7th row: the result combined with HLS S channel, HSV V channel, gradient on x and y direction.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `cv2.warpPerspective()`, which appears in lines 151 through 172 in the file `UndistAndPerspTransf.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
![alt text][image4]
```python
src = np.float32([[570,468],  [714,468], [1106,720], [207,720]])
dst = np.float32([[320,1],[920,1],[920,720],[320,720]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 468      | 320, 1        | 
| 714,468       | 920,1         |
| 1106,720      | 960, 720      |
| 207,720       | 320,720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a straight line test image, whose warped image has straight and parallel lines. 
![alt text][image6]
And the warped counterparts for tilted lane-lines appear parallel.
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are several approaches to map out the lane-lines: using the histogram and using convolution, for example, which both use sliding windows to progress from the bottom to the top of image.
I used the histogram of the bottom half image per column to initialize the search for the line locations.

![alt text][image7]
A number of windows are set to search the lines vertically from the bottom upwards. The windows have same widths and same height. Each window gets its location updated by searching the area above the previous window. The new window's horizontal midpoint is an average of the non zero points within the previous window's margin.
The number of windows is adjusted for the resolution of the tracking, and the margin of window is adjusted as well. Its result is shown with the following step.
The curves of each line is fit by a second order polynomial from the points identified for each line: f(y) = Ay^2^+By+C
```
# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
![alt text][image8] 
![alt text][image9]
For test2 image: 
The fitted curve for left line: Y~left~= 0.0001335*y^2^ - 0.1906*y + 307.44
The fitted curve for right line: Y~right~= 0.0003379*y^2^ - 0.3054*y + 891.73

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature at point x is: R~curve~ = ((1+(2Ay+B)^2^)^1.5^/|2A|
With calculated A,B,C from the previous step, the curvature is implemented as (line 171-199 in findingLanesHist.py):
```
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)
```
To get the curvature in the real world instead of the image, an estimation of distance ratio from image to the world is used:
```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
The polynomial fitting is computed again with data in the world space, and so are the curvatures. 
- The curvature for test2 is: left 1230.46982432m and right: 1486.342089683 m

Now compute the distance between the middle of the vehicle and the center of the lane. The camera is installed in the middle of the car, so the car's position is in the center (bottom) of the image. The car is at the bottom of the image, so image height is applied to the fitting polynomials to get the two lane line positions in the image. The lane center is in the middle of the two lines. The distance in the image is transformed into the world space by again `xm_per_pix` = 3.7/700.
- The distance of the car to the lane center is 0.084 m in test2.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 208 through 234 in my code in `findingLanesHist.py` in the function `map_lane()`.  Here is an example of my result on a test image:
```
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
undist = cv2.imread('./test_images/undistorted_test2.jpg')
undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
height, width, chann = undist.shape
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
```
![alt text][image10]

---

### Pipeline (video)
Now in order to generate images of each frame in a video, I put all scripts as functions. Camera calibration data and perspective transform parameters are loaded from file by pickle.
* A Line class is used to remember a few numbers of previous fit, and current best fit of both lines. An update() function is added to the class, to update the best fit.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Although the current combination of thresholding on gradient x, y, and S and V channel work stably for project video, it fails at Challenge video when the car is under the bridge. Other color space / channel should be examined.

![alt text][image11]

I spent a lot of time in locating the source points for perspective transformation. Later I realized I could use x and y location instead of percentage of the image. But this does not seem very generalizable.
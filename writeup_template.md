##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/features.png
[image3]: ./output_images/norm_features.png
[image4]: ./output_images/windows_scaled.png
[image5]: ./output_images/find_cars_test1.png
[image6]: ./output_images/find_cars_test4.png
[image7]: ./output_images/figure_2.png
[image8]: ./output_images/figure_3.png
[image9]: ./output_images/figure_4.png
[image10]: ./output_images/figure_5.png
[image11]: ./output_images/figure_6.png
[image12]: ./output_images/figure_7.png
[image13]: ./output_images/test1.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Main Files (python):
* lesson_functions.py - Functions provided throughout the project lesson
* helper_functions.py - Extended lession_functions to provide remaining pipeline functionality
* process_data.py - Loaded labeled data and extracted features to create training & test data
* data_structure.py - Data Structure class used during vehicle finding pipeline
* P5Video.py - Usage: P5Video.py <input_file> <output_file>

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The main code for this step is contained in the function `extract_features`, lines 73 through 104 of the file called `lesson_functions.py`. It is called by function `readData` in lines 34 through 38, of the file called `process_data.py`.  

`readData` starts by reading in all the `vehicle` and `non-vehicle` images from GTI, KITTI, etc.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The training image file names, extracted features, and feature parameters were stored in a data structure through function `setup`, lines 94 through 116 in process_data.py. `setup` returns this data structure as well as a trained classifier, SVM (discussed below)

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that a spatial binning size of (16, 16) out-performed (32, 32) while the remainder were mostly set from the project lessons.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 3 types of features: 1). spatial color features, 2). color histogram and 3) hog features from all channels. Here is an image showing the normalized feature vector combined from all three types:

![alt text][image3]

The main code for this step is contained in the function `train_SVM_LinearSVC`, lines 38 through 63 of the file called `helper_functions.py`. I also explored using decision trees and include a similar function, `train_decision_tree`, lines 13 through 36 of the file called `helper_functions.py`, which is not used in the pipeline. In a head to head comparison with SVM I found that the decision tree classifier was less accurate and required more processing time. Here is a comparison of such results using spatial bins (32,32):

SVM:
 * color, spatial: Test Accuracy of SVC = 0.9181, 0.01135 Seconds to predict 10 labels with SVC
 * color, spatial, hog0: Test Accuracy of SVC = 0.9716, 0.01754 Seconds to predict 10 labels with SVC
 * color, spatial, hogAll: Test Accuracy of SVC = 0.9797, 0.0286 Seconds to predict 10 labels with SVC

Decision Tree: 
 * color, spatial: Test Accuracy of DT = 0.9077, 0.01928 Seconds to predict 10 labels with DT
 * color, spatial, hog0: Test Accuracy of DT = 0.9223, 0.03695 Seconds to predict 10 labels with DT
 * color, spatial, hogAll: Test Accuracy of DT = 0.92, 0.05914 Seconds to predict 10 labels with DT

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The main code for this step is contained in the function `find_cars_scaled`, lines 128 through 177 of the file called `helper_functions.py`. It calls function `find_cars` in lines 309 through 375, of the file called `lesson_functions.py` which was provided by the lesson to extract features using hog sub-sampling. My additions to the `find_cars_scaled` function includes an array of scales, for creating windows, as well as a threshold for heat-map filtering.

The code to determine parameters for the sliding window search can be found in exploratory function, `slide_window_scaled`, lines 112 through 126 of the file called `helper_functions.py`. Here's a sample image where I overlaid the output boxes on several image frames before deciding on the range of scale.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales [1, 1.5, 2, 2.5] using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided results with false-positives. HSV worked well (accuracy .99) since the vehicles are fairly saturated in color, compared to the background and YCrCb performed comparably. Here are some example images:

![alt text][image5]
![alt text][image6]
---

To optimize the performance of my classifier I tried different combinations of the parameters listed above and used the following final parameters that resulted in .99 accuracy performance. I also used all labeled data examples provided to conduct training and performance evaluzation. The training data set avoids a class imbalance issue being of 8792 car and 8968 not car images. To avoid video sequence bias, this dataset was randomized and split into 80% training and 20% test data (lines 78 through 81 of helper_functions.py)

### Final parameters:
* color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 9  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block
* hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
* spatial_size = (16, 16) # Spatial binning dimensions
* hist_bins = 16    # Number of histogram bins
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off
* y_start_stop = [None, None] # Min and max in y to search in slide_window()
* hist_range=(0, 256)

### Final performance:
* Feature vector length: 6108
* Number of training samples :  14208
* Number of test samples :  3552
* 19.15 Seconds to train SVM Linear SVC...
* Test Accuracy of SVC =  0.9885

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The main code for filtering false positives is contained in the class `dataStructure`, lines 25 through 38 of the file called `data_structure.py`. A list of bounding boxes, of a specified running average size `n`, is maintained across multiple frames. Upon the insertion of a new list of bounding boxes from the current frame, function `insert_bbox_list` returns a flattened, list combining bounding boxes from `n` frames.

The main code for combinng overlapping bounding boxes is contained in the function `find_cars_scaled`, lines 139 through 156 of the file called `helper_functions.py`. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding bounding boxes, heatmaps, label and detection:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### Here is the previous image from section 2. after heat map and filtering for false-positives:

![alt text][image13]
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

The speed of performance can definitely be improved. For example, improvements can come from multi-scaled window being called on a single instance of HOG feature generation. The pipeline still occasionally have false-positives. To be more robust we can leverage Advanced-Lane-Finding project and restrict searches in lanes going in the same direction. An increase in sample data, and better resolution would also result in more robust classifiers. This may have been provided in augmented dataset from udacity but wasn't combined into this submission.  


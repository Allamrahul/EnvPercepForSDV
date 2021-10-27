# EnvPercepForSDV

The goal of this project is to extract useful scene information to allow self-driving cars to safely and reliably traverse their environment.

In this project, I have used the output of semantic segmentation neural networks to implement drivable space estimation in 3D, lane estimation, filter errors in the output of 2D object detectors and determine how far obstacles are from the self-driving car.

We first implement drivable space estimation in 3D. We are given the output of a semantic segmentation neural network, the camera calibration matrix K, as well as the depth per pixel. In the context of self-driving cars, drivable space includes any space that the car is physically capable of traversing in 3D. The task of estimating the drivable space is equivalent to estimating pixels belonging to the ground plane in the scene. We used RANSAC to estimate the ground plane in the 3D camera coordinate frame from the x,y, and z coordinates estimated above.

Next, we do lane Estimation Using The Semantic Segmentation Output

We use the output of semantic segmentation to estimate the lane boundaries of the current lane the self-driving car is using. This task can be separated to two subtasks, lane line estimation, and post-processing through horizontal line filtering and similar line merging. The first step to perform this task is to estimate any line that qualifies as a lane boundary using the output from semantic segmentation. The second subtask to perform the estimation of the current lane boundary is to merge redundant lines, and filter out any horizontal lines apparent in the image. Merging redundant lines can be solved through grouping lines with similar slope and intercept. Horizontal lines can be filtered out through slope thresholding.

Next, we Computing Minimum Distance To Impact Using The Output of 2D Object Detection

For this, we use 2D object detection output to determine the minimum distance to impact with objects in the scene. However, the task is complicated by the fact that the provided 2D detections are from a high recall, low precision 2D object detector. We will first be using the semantic segmentation output to determine which bounding boxes are valid. Then, we will compute the minimum distance to impact using the remaining bounding boxes and the depth image.

Finally, we filter out Unreliable Detections using the semantic segmentation output

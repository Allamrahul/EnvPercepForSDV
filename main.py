
# # Environment Perception For Self-Driving Cars
#
# - Use the output of semantic segmentation neural networks to implement drivable space estimation in 3D.
# - Use the output of semantic segmentation neural networks to implement lane estimation.
# - Use the output of semantic segmentation to filter errors in the output of 2D object detectors. 
# - Use the filtered 2D object detection results to determine how far obstacles are from the self-driving car.

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *

np.random.seed(1)
np.set_printoptions(precision=2, threshold=np.nan)


# ## 0 - Loading and Visualizing the Data
def main():

    dataset_handler = DatasetHandler()


    # The dataset handler contains three test data frames 0, 1, and 2. Each frames contains:
    # - DatasetHandler().rgb: a camera RGB image
    # - DatasetHandler().depth: a depth image containing the depth in meters for every pixel.
    # - DatasetHandler().segmentation: an image containing the output of a semantic segmentation neural network as the category per pixel.
    # - DatasetHandler().object_detection: a numpy array containing the output of an object detection network.
    #

    image = dataset_handler.image
    plt.imshow(image)

    k = dataset_handler.k
    print(k)

    depth = dataset_handler.depth
    plt.imshow(depth, cmap='jet')


    # The semantic segmentation output can be accessed in a similar manner through:

    segmentation = dataset_handler.segmentation
    plt.imshow(segmentation)


    # ### Segmentation Category Mappings:
    # The output segmentation image contains mapping indices from every pixel to a road scene category. To visualize the semantic segmentation output, we map the mapping indices to different colors. The mapping indices and visualization colors for every road scene category can be found in the following table:
    #
    # |Category |Mapping Index| Visualization Color|
    # | --- | --- | --- |
    # | Background | 0 | Black |
    # | Buildings | 1 | Red |
    # | Pedestrians | 4 | Teal |
    # | Poles | 5 | White |
    # | Lane Markings | 6| Purple |
    # | Roads | 7 | Blue |
    # | Side Walks| 8 | Yellow |
    # | Vehicles| 10 | Green |
    #
    # The vis_segmentation function of the dataset handler transforms the index image to a color image for visualization:

    colored_segmentation = dataset_handler.vis_segmentation(segmentation)

    plt.imshow(colored_segmentation)

    # The set_frame function takes as an input a frame number from 0 to 2 and loads that frame allong with all its associated data. It will be useful for testing and submission at the end of this assesment.

    dataset_handler.set_frame(2)

    image = dataset_handler.image
    plt.imshow(image)

    # ### - Estimating The Ground Plane Using RANSAC:
    #
    # In the context of self-driving cars, drivable space includes any space that the car is physically capable of traversing in 3D. The task of estimating the drivable space is equivalent to estimating pixels belonging to the ground plane in the scene.
    # We will use RANSAC to estimate the ground plane in the 3D camera coordinate frame from the x,y, and z coordinates estimated above.
    #
    # The first step is to process the semantic segmentation output to extract the relevant pixels belonging to the class you want consider as ground. For this assessment, that class is the road class with a mapping index of 7.

    # Get road mask by choosing pixels in segmentation output with value 7
    road_mask = np.zeros(segmentation.shape)
    road_mask[segmentation == 7] = 1

    # Show road mask
    plt.imshow(road_mask)

    # Get x,y, and z coordinates of pixels in road mask
    x_ground = x[road_mask == 1]
    y_ground = y[road_mask == 1]
    z_ground = dataset_handler.depth[road_mask == 1]
    xyz_ground = np.stack((x_ground, y_ground, z_ground))

    p_final = ransac_plane_fit(xyz_ground)
    print('Ground Plane: ' + str(p_final))

    dist = np.abs(dist_to_plane(p_final, x, y, z))

    ground_mask = np.zeros(dist.shape)

    ground_mask[dist < 0.1] = 1
    ground_mask[dist > 0.1] = 0

    plt.imshow(ground_mask)

    dataset_handler.plot_free_space(ground_mask)

    # However, estimating the drivable space is not enough for the self-driving car to get on roads. The self-driving car still needs to perform lane estimation to know where it is legally allowed to drive. Once you are comfortable with the estimated drivable space, continue the assessment to estimate the lane where the car can drive.
    #
    # ## - Lane Estimation Using The Semantic Segmentation Output
    #
    # use the output of semantic segmentation to estimate the lane boundaries of the current lane the self-driving car is using. This task can be separated to two subtasks, lane line estimation, and post-processing through horizontal line filtering and similar line merging.
    #
    # Estimating Lane Boundary Proposals:
    # The first step to perform this task is to estimate any line that qualifies as a lane boundary using the output from semantic segmentation. We call these lines 'proposals'.
    #

    lane_lines = estimate_lane_lines(segmentation)

    print(lane_lines.shape)

    plt.imshow(dataset_handler.vis_lanes(lane_lines))

    merged_lane_lines = merge_lane_lines(lane_lines)

    plt.imshow(dataset_handler.vis_lanes(merged_lane_lines))

    max_y = dataset_handler.image.shape[0]
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

    extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
    final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
    plt.imshow(dataset_handler.vis_lanes(final_lanes))

    detections = dataset_handler.object_detection

    plt.imshow(dataset_handler.vis_object_detection(detections))

    # Detections have the format [category, x_min, y_min, x_max, y_max, score]. The Category is a string signifying the classification of the bounding box such as 'Car', 'Pedestrian' or 'Cyclist'. [x_min,y_min] are the coordinates of the top left corner, and [x_max,y_max] are the coordinates of the bottom right corners of the objects. The score signifies the output of the softmax from the neural network.

    print(detections)

    filtered_detections = filter_detections_by_segmentation(detections, segmentation)

    plt.imshow(dataset_handler.vis_object_detection(filtered_detections))

    min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

    print('Minimum distance to impact is: ' + str(min_distances))

    font = {'family': 'serif', 'color': 'red', 'weight': 'normal', 'size': 12}

    im_out = dataset_handler.vis_object_detection(filtered_detections)

    for detection, min_distance in zip(filtered_detections, min_distances):
        bounding_box = np.asfarray(detection[1:5])
        plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m',
                 fontdict=font)

    plt.imshow(im_out)


# ##  Drivable Space Estimation Using Semantic Segmentation Output
# 
# Implementing drivable space estimation in 3D. You are given the output of a semantic segmentation neural network, the camera calibration matrix K, as well as the depth per pixel.
# 
# ###  Estimating the x, y, and z coordinates of every pixel in the image:

# xy_from_depth
def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """

    # Get the shape of the depth tensor
    sz = depth.shape
    # Grab required parameters from the K matrix
    f = k[0,0]
    C_u = k[0,2]
    C_v = k[1,2]

    # Generate a grid of coordinates corresponding to the shape of the depth map
    u,v = np.meshgrid(np.arange(1,sz[1]+1,1), np.arange(1,sz[0]+1,1))
    # Compute x and y coordinates
    
    x = ((u - C_u) * depth)/f
    y = ((v - C_v) * depth)/f
    
    return x, y

# The next step is to use the extracted x, y, and z coordinates of pixels belonging to the road to estimate the ground plane. RANSAC will be used for robustness against outliers.

# Implementing RANSAC for plane estimation. steps:
# 1. Choose a minimum of 3 points from xyz_ground at random.
# 2. Compute the ground plane model using the chosen random points, and the provided function compute_plane.
# 3. Compute the distance from the ground plane model to every point in xyz_ground, and compute the number of inliers based on a distance threshold.
# 4. Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.  
# 5. Repeat until number of iterations $\geq$ a preset number of iterations, or number of inliers minimum number of inliers.
# 6. Recompute and return a plane model using all inliers in the final inlier set. 
#
# 
# 
# RANSAC Plane Fitting

def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    # Set thresholds:
    num_itr = 100  # RANSAC maximum number of iterations
    min_num_inliers = 45000  # RANSAC minimum number of inliers
    distance_threshold = 0.1  # Maximum distance from point to plane for point to be considered inlier
    
    data_size = xyz_data.shape[1]
    x = xyz_data[0,:]
    y = xyz_data[1,:]
    z = xyz_data[2,:]
    
    max_inliers = 0

    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        inds = np.random.choice(data_size, 15, replace = False)

        # Step 2: Compute plane model
        p = compute_plane(xyz_data[:,inds])
        
        # Step 3: Find number of inliers
        dist = np.abs(dist_to_plane(p,x,y,z))
        n_inliers = np.sum(dist < distance_threshold)

        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if max_inliers < n_inliers:
            max_inliers = n_inliers
            inlier_set = xyz_data[:,dist < distance_threshold]
        
        # Step 5: Check if stopping criterion is satisfied and break.         
        if max_inliers > min_num_inliers:
            break
        
    # Step 6: Recompute the model parameters using largest inlier set.
    p = compute_plane(inlier_set)
    
    return p

# Estimate lane line proposals. Here are the 3 steps:
# 1. Create an image containing the semantic segmentation pixels belonging to categories relevant to the lane boundaries, similar to what we have done previously for the road plane. For this assessment, these pixels have the value of 6 and 8 in the neural network segmentation output.
# 2. Perform edge detection on the derived lane boundary image.
# 3. Perform line estimation on the output of edge detection.

# estimate_lane_lines


def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """

    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    mask = np.zeros_like(segmentation_output, dtype=np.uint8)
    mask[segmentation_output == 6] = 255
    mask[segmentation_output == 8] = 255

    # Step 2: Perform Edge Detection using cv2.Canny()
    edges = cv2.Canny(mask, 50, 150)

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    params = dict(
        rho=10,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=50
    )
    lines = cv2.HoughLinesP(edges, **params)

    return lines.reshape((-1, 4))


# ### Merging and Filtering Lane Lines:
# 
# The second subtask to perform the estimation of the current lane boundary is to merge redundant lines, and filter out any horizontal lines apparent in the image. Merging redundant lines can be solved through grouping lines with similar slope and intercept. Horizontal lines can be filtered out through slope thresholding. 
# 
# Post-process the output of the function ``estimate_lane_lines`` to merge similar lines, and filter out horizontal lines using the slope and the intercept. The three steps are:
# 1. Get every line's slope and intercept using the function provided.
# 2. Determine lines with slope less than horizontal slope threshold. Filtering can be performed later if needed.
# 3. Cluster lines based on slope and intercept as you learned in Module 6 of the course. 
# 4. Merge all lines in clusters using mean averaging.

# merge_lane_lines
def merge_lane_lines(
        lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """

    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    clusters = []
    current_inds = []
    itr = 0

    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)

    # Step 2: Determine lines with slope less than horizontal slope threshold.
    slopes_horizontal = np.abs(slopes) > min_slope_threshold

    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        in_clusters = np.array([itr in current for current in current_inds])
        if not in_clusters.any():
            slope_cluster = np.logical_and(slopes < (slope + slope_similarity_threshold),
                                           slopes > (slope - slope_similarity_threshold))
            intercept_cluster = np.logical_and(intercepts < (intercept + intercept_similarity_threshold),
                                               intercepts > (intercept - intercept_similarity_threshold))
            inds = np.argwhere(slope_cluster & intercept_cluster & slopes_horizontal).T
            if inds.size:
                current_inds.append(inds.flatten())
                clusters.append(lines[inds])
        itr += 1

    # Step 4: Merge all lines in clusters using mean averaging
    merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    merged_lines = np.array(merged_lines).reshape((-1, 4))


    return merged_lines

# ### - Filtering Out Unreliable Detections:
# The first thing you can notice is that an wrong detection occures on the right side of the image. What is interestingis that this wrong detection has a high output score of 0.76 for being a car. Furthermore, two bounding boxes are assigned to the vehicle to the left of the image, both with a very high score, greater than 0.9. This behaviour is expected from a high precision, low recall object detector. To solve this problem, the output of the semantic segmentation network has to be used to eliminate unreliable detections.
# 
#  Eliminate unreliable detections using the output of semantic segmentation. The three steps are:
# 1. For each detection, compute how many pixels in the bounding box belong to the category predicted by the neural network.
# 2. Devide the computed number of pixels by the area of the bounding box (total number of pixels).
# 3. If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.


# filter_detections_by_segmentation
def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.
    
    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    
    # Set ratio threshold:
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    filtered_detections = []
    for detection in detections:
        
        # Step 1: Compute number of pixels belonging to the category for every detection.
        x_min = int(float(detection[1])) # ew
        y_min = int(float(detection[2]))
        x_max = int(float(detection[3]))
        y_max = int(float(detection[4]))
        correct_pixels = (segmentation_output[y_min:y_max, x_min:x_max] == 10).sum()
        num_pixels = (x_max - x_min) * (y_max - y_min)
        
        # Step 2: Devide the computed number of pixels by the area of the bounding box (total number of pixels).
        ratio = correct_pixels / num_pixels
        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if ratio > ratio_threshold:
            filtered_detections.append(detection)
    
    return filtered_detections


# ### Estimating Minimum Distance To Impact:
# 
# estimate the minimum distance to every bounding box in the input detections. This can be performed by simply taking the minimum distance from the pixels in the bounding box to the camera center.
# 
# Compute the minimum distance to impact between every object remaining after filtering and the self-driving car. The two steps are:
# 
# 1. Compute the distance to the camera center using the x,y,z arrays from  part I. This can be done according to the equation: $ distance = \sqrt{x^2 + y^2 + z^2}$.
# 2. Find the value of the minimum distance of all pixels inside the bounding box.


# find_min_distance_to_detection:
def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """

    min_distances = []
    for detection in detections:
        # Step 1: Compute distance of every pixel in the detection bounds
        x_min = int(float(detection[1]))
        y_min = int(float(detection[2]))
        x_max = int(float(detection[3]))
        y_max = int(float(detection[4]))
        min_dist = np.inf
        
        for u in range(y_min, y_max):
            for v in range(x_min, x_max):
                dist = np.sqrt(x[u, v]**2 + y[u, v]**2 + z[u, v]**2)
                if dist < min_dist:
                    min_dist = dist
        min_distances.append(min_dist)

    return min_distances


def validate():

    dataset_handler = DatasetHandler()
    dataset_handler.set_frame(1)
    segmentation = dataset_handler.segmentation
    detections = dataset_handler.object_detection
    z = dataset_handler.depth

    # Part 1
    k = dataset_handler.k
    x, y = xy_from_depth(z, k)
    road_mask = np.zeros(segmentation.shape)
    road_mask[segmentation == 7] = 1
    x_ground = x[road_mask == 1]
    y_ground = y[road_mask == 1]
    z_ground = dataset_handler.depth[road_mask == 1]
    xyz_ground = np.stack((x_ground, y_ground, z_ground))
    p_final = ransac_plane_fit(xyz_ground)

    # Part II
    lane_lines = estimate_lane_lines(segmentation)
    merged_lane_lines = merge_lane_lines(lane_lines)
    max_y = dataset_handler.image.shape[0]
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

    extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
    final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)

    # Part III
    filtered_detections = filter_detections_by_segmentation(detections, segmentation)
    min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

    # Print Submission Info

    final_lane_printed = [list(np.round(lane)) for lane in final_lanes]
    print('plane:')
    print(list(np.round(p_final, 2)))
    print('\n lanes:')
    print(final_lane_printed)
    print('\n min_distance')
    print(list(np.round(min_distances, 2)))


    # ### Visualize your Results:
    #
    # Make sure your results visualization is appealing before submitting your results.


    # Original Image
    plt.imshow(dataset_handler.image)


    # Part I
    dist = np.abs(dist_to_plane(p_final, x, y, z))

    ground_mask = np.zeros(dist.shape)

    ground_mask[dist < 0.1] = 1
    ground_mask[dist > 0.1] = 0

    plt.imshow(ground_mask)


    # Part II
    plt.imshow(dataset_handler.vis_lanes(final_lanes))



    # Part III
    font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

    im_out = dataset_handler.vis_object_detection(filtered_detections)

    for detection, min_distance in zip(filtered_detections, min_distances):
        bounding_box = np.asfarray(detection[1:5])
        plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)

    plt.imshow(im_out)

if __name__ == "__main__":
    main()
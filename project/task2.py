"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random
import math

"""
@author :   Syed Rehman
            CSE-473
            proj2 task-2
	    syedrehm@buffalo.edu
"""

def trim_sides(frame):
    #recursively crop black sidebars
    if not np.sum(frame[0]):
        return trim_sides(frame[1:])
    if not np.sum(frame[-1]):
        return trim_sides(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim_sides(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim_sides(frame[:,:-2])
    return frame



def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    # right side must always be image on the right side.. or else results wont work!
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)      #good side
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)    #bad side always !! 

    
    #calculate the derivative in respect to x then y.
    """
    left_dx, left_dy = np.gradient(left_gray)
    right_dx, right_dy = np.gradient(right_gray)
    
    |~        ~|
    |Ix^2  Ixy |
    |Ixy   Iy^2|
    |~        ~|
    
    left_Ixx = left_dx * left_dx
    left_Iyy = left_dy * left_dy
    left_Ixy = left_dx * left_dy
    
    right_Ixx = right_dx * right_dx
    right_Iyy = right_dy * right_dy
    right_Ixy = right_dx * right_dy
    """

    sift = cv2.xfeatures2d.SIFT_create()
    
    left_key_points, left_descriptors = sift.detectAndCompute(left_gray, None)
    right_key_points , right_descriptors = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher()
    # matches 'bad' image thats turned to the good one.
    points_matches = bf.knnMatch(right_descriptors,left_descriptors, k=2)   # (bad , good)

    pick = []
    for i,j in points_matches:
        if i.distance < 0.8 * j.distance:
            pick.append(i)
    
    #   queryIdx refers to keypoints of the bad side or right in our case 
    #   and trainIdx refers to keypoints on the left
    src_ = np.float32([right_key_points[m.queryIdx].pt for m in pick]).reshape(-1, 1, 2)    
    dest_ = np.float32([left_key_points[m.trainIdx].pt for m in pick]).reshape(-1, 1, 2)

    #source to destination or (bad side, good side, cv2.RANSAC, 5.0)
    Homography_matrix, mask = cv2.findHomography(src_, dest_, cv2.RANSAC, 5.0)  

    height, width = right_gray.shape    #for the bad side
    pts = np.float32([ [0,0],[0, height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    #TR matrix for the right side.
    output_img = cv2.perspectiveTransform(pts, Homography_matrix)

    left_gray = cv2.polylines(left_gray,[np.int32(output_img)],True,255,3, cv2.LINE_AA)

    output_img = cv2.warpPerspective(right_img, Homography_matrix,(left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    
    output_img[0:left_img.shape[0],0:right_img.shape[1]] = left_img


    return trim_sides(output_img)
    # raise NotImplementedError

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)

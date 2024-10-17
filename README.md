# Image Segmentation using K-Means and DBSCAN

This project implements two clustering algorithms, K-Means and DBSCAN, from scratch to perform segmentation of color images. Each pixel in the image is represented as a 3-dimensional vector in the RGB color space, and the goal is to segment the image by clustering these pixel vectors.

## Features
- *Clustering Algorithms*:
  - *K-Means Clustering*: Allows the user to specify the number of clusters k. The algorithm iterates to assign each pixel to one of k clusters based on its RGB vector.
  - *DBSCAN (Density-Based Spatial Clustering of Applications with Noise)*: Allows the user to specify the parameters ε (epsilon) and minPts. DBSCAN identifies clusters based on density, grouping together pixels that are closely packed while labeling others as noise.

## Input/Output
- *Input*: A non-segmented color image.
- *Output*: A segmented version of the input image, where pixels belonging to the same cluster are represented by the same color.

## Parameters
- For *K-Means*:
  - k: The number of clusters.
  
- For *DBSCAN*:
  - ε: The maximum distance between two samples for them to be considered part of the same neighborhood.
  - minPts: The minimum number of points required to form a dense region (cluster).



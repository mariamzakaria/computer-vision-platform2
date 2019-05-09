
# 1.Corner detection

## Harris operator


## Steps:

##           1.Color image to Grayscale conversion 
##            2.image smoothing by convoluion and gaussian kernel
##            3.compute Gradient_x and Gradient_y by Sobel kernel
##            4.compute the Hessian matrix
##            5.compute Eigen values of Hessian matrix 
##            6.evaluate corners and edges using R as a measure
##                 R=(λ1×λ2)−k(λ1+λ2)
##             . Edge : R < 0
##             .Corner : R > 0


<img src = "/images/corner2.PNG" width = "50%">

<img src = "/images/corner1.PNG" width = "50%">

<img src = "/images/corner3.PNG" width = "50%">

<img src = "/images/corner4.PNG" width = "50%">

# Detect Lines (HoughLines)
https://gist.github.com/rishabhsixfeet/45cb32dd5c1485e273ab81468e531f09
## Steps:

##            1.Color image to Grayscale conversion 
##            2.Get a binary edge image by canny edge detector
##            3.build function for hough accumulator
##            4.drawing the lines from the hough accumulatorlines
##            5.run hough accumulatorlines on the canny edge image

<img src = "/images/detectedlines.PNG" width = "50%">

<img src = "/images/detectedlines2.PNG" width = "50%">

<img src = "/images/detectedlines3.PNG" width = "50%">

# Segmentation and Clustring

# 1- Using Region growing 

## this links help us: https://github.com/suhas-nithyanand/Image-Segmentation-using-Region-Growing 

## https://github.com/A-Alaa/cv-assignments/blob/master/scripts/RegionGrowingSegmentation.py 



## 1. convert image to gray scale
## 2. using cv.mouseEvent  to choose a region for segmentation, we just choose a  point then we get the  Neighbors points to be segmented, If threshold is active all values less than this value will be ignored



<img src = "/images/regiongrowing.PNG" width = "50%">

## we couldn't apply mouseEvent in the figure of outputimage in the gui, so we use openCV for showing results and using mouseEvent

# 2- using Kmeans

## We set unmber of itrations =5 to get good result
## 1. Initialise data vector with attribute r,g,b,x,y for each pixel in input image
## 2. Initialise vector that holds which cluster a pixel is currently in 
## 3. Standarize the values of our features
## 4. Set pixels to their cluster
## 5. Check if a cluster is ever empty, if so append a random datapoint to it contains an array with all clusters, [True True False True * n of clusters] False means empty 
## 6. set centers then Move centers to the centroid of their cluster 
## 7. set the pixels on original image to be that of the pixel's cluster's centroid


<img src = "/images/Kmeans.PNG" width = "50%">

### no of iterations = 5 
## We couldn't set the output image to the gui directly so we save it the show it in gui
## this link help us to make Kmeans: https://github.com/asselinpaul/ImageSeg-KMeans 

# 3- MeanShift 

## 1. making color space of the input image
## 2. making num of clusters according to the input image
## 3. making function iterate in the given window indices, to find its center of mass
## 4. making function classify the image component based on the its value

## this link helped us : https://github.com/A-Alaa/cv-assignments/blob/master/scripts/hisMeanShift.py

<img src = "/images/meanShift.PNG" width = "50%">

## we can't show the output image in the gui so we use opencv to show it 

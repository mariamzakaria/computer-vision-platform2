
from PyQt5 import QtCore, QtGui, QtWidgets,uic
import os
import sys
sys.path.insert(0, "C:\\Users\\medoz\\Anaconda3\\Lib\\site-packages")
sys.path.insert(0, "C:\\Users\\medoz\\Anaconda3\\lib\\site-packages")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.image as mpimg
from matplotlib import cm
from math import sqrt, pi, cos, sin, atan2
from PIL import Image, ImageDraw
from collections import defaultdict
import  qimage2ndarray
from skimage.feature import canny
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from  matplotlib.backends.backend_qt5agg  import  FigureCanvas
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from  matplotlib.figure  import  Figure

from scipy import signal
from scipy import ndimage
import pyqtgraph as pg

import snake as sn
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


#CONVER TO GRAYSCALE
def rgb2gray(rgb_image):
    return 0.299*rgb_image[:,:,0]+0.587*rgb_image[:,:,1]+0.114*rgb_image[:,:,2]
    

#Box Filter
def box_filter( w ):
    return np.ones((w,w)) / (w*w)

#Gaussian Filter
def gaussian_kernel( kernlen , std ):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

#Median Filter
def salt_n_pepper(img):
    salt_pepper = np.random.random(img.shape) * 255
    pepper = salt_pepper < 30
    salt = salt_pepper > 225;img[pepper] = 0
    img[salt] = 255
    return img
    


def convetToGray():
     dig.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
     if dig.fileName:
         color_image = Image.open(dig.fileName)
         image = mpimg.imread(dig.fileName)
        # bw = np.dot(color_image[...,:3], [0.299, 0.587, 0.114])
         bw = color_image.convert('L')
         bw.save('grayimage.bmp')
          
def plotoutput(img):
    yourQImage=qimage2ndarray.array2qimage(img)
    gray=QtGui.QImage(yourQImage)
    pixmap  = QtGui.QPixmap.fromImage(gray)
    pixmap = pixmap.scaled(dig.label_filters_output.width(), dig.label_filters_output.height(), QtCore.Qt.KeepAspectRatio)
    dig.label_filters_output.setPixmap( pixmap) # Set the pixmap onto the label
    dig.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)
    
def plotinput(img):
    yourQImage=qimage2ndarray.array2qimage(img)
    gray=QtGui.QImage(yourQImage)
    pixmap  = QtGui.QPixmap.fromImage(gray)
    pixmap = pixmap.scaled(dig.label_filters_input.width(), dig.label_filters_input.height(), QtCore.Qt.KeepAspectRatio)
    dig.label_filters_input.setPixmap( pixmap) # Set the pixmap onto the label
    dig.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)           
          
def setImageFT():
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
    if fileName: # If the user gives a file
        dig.image= mpimg.imread( fileName)
        plotinput(dig.image)
        dig.valueChannel = extractValueChannel(dig.image)
        plotoutput(dig.valueChannel)
              
#Browse Images    
def setImage():
  
    dig.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
    if dig.fileName:
        image= mpimg.imread(dig.fileName)
        dig.image=rgb2gray(image)
        plotinput(dig.image)

# DETECT LINES
def HoughLines():
  
    dig.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
    if dig.fileName:
        dig.image= Image.open(dig.fileName)
        dig.pixmap = QtGui.QPixmap(dig.fileName) # Setup pixmap with the provided image
        dig.pixmap = dig.pixmap.scaled(dig.label_lines_input.width(), dig.label_lines_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        dig.label_lines_input.setPixmap(dig.pixmap) 
        
        Shapeee = np.array(dig.image)
        edges = canny(Shapeee, 2, 1, 25)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(dig.image, cmap=cm.gray)
        ax[0].set_title('Input image')


        ax[1].imshow(edges, cmap=cm.gray)
        ax[1].set_title('Canny edges')
        #edges.save("mmmmmm.bmp")

        x = edges * 0
        

        ax[2].imshow(x)
        output_image1 = Image.new("RGB", dig.image.size)
        draw = ImageDraw.Draw(output_image1)
        for line in lines:
            p0, p1 = line
            draw.point((p0,p1),(255,255,255))

            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))

        ax[2].set_title('Probabilistic Hough')
        #x.imsave("Hough.bmp",imgf)
        
       # scipy.misc.imsave('outfilerrr.bmp', edges * 0)
        output_image1.save("cannyline.bmp")
        dig.pixma = QtGui.QPixmap("cannyline.bmp") # Setup pixmap with the provided image
        dig.pixma = dig.pixma.scaled(dig.label_lines_input_2.width(), dig.label_lines_input_2.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        dig.label_lines_input_2.setPixmap(dig.pixma)
        dig.label_lines_input_2.setAlignment(QtCore.Qt.AlignCenter) 

      

# Apply filters on images
def setFilters(text):
#prewit FILTER
    if dig.comboBox.currentIndex() == 1:
        prewit(dig.image)
#SOBEL FILTER
    if dig.comboBox.currentIndex() == 2:
        sobel(dig.image)
#laplacian FILTER
    if dig.comboBox.currentIndex() == 3:
        noisy1=addnoise(dig.image)
        plotinput(noisy1)
        laplacian(dig.image)
#laplacian of gaussian FILTER
    if dig.comboBox.currentIndex() == 4 :
        noisy2=addnoise(dig.image)
        plotinput(noisy2)
        log(noisy2)         
#difference of gaussian Filter    
    if dig.comboBox.currentIndex() == 5:
        noisy=addnoise(dig.image)
        plotinput(noisy)
        dog(noisy)


#BOX FILTER        
    if dig.comboBox.currentIndex() == 6:

        noisy=salt_n_pepper(dig.image)
        plotinput(salt_n_pepper(dig.image))
 
        filtered_img_box9 = signal.convolve2d(noisy, box_filter(9) ,'same')
        plotoutput(filtered_img_box9)

#GAUSSIAN FILTER 
    if dig.comboBox.currentIndex() == 7:

        noisy=salt_n_pepper(dig.image)
        plotinput(salt_n_pepper(dig.image))

        filtered_img_g7_std10 = signal.convolve2d(noisy, gaussian_kernel(7,.3) ,'same')
        plotoutput(filtered_img_g7_std10 )

#MEDIAN FILTER               
    if dig.comboBox.currentIndex() == 8:
  

        noisy=salt_n_pepper(dig.image)
        plotinput(salt_n_pepper(dig.image))

        med_image3 = ndimage.median_filter(noisy,(3,3))
        plotoutput(med_image3 )

 #sharpen FILTER
    if dig.comboBox.currentIndex() == 9:
        sharpen(dig.image)

    
#sharpen FILTER
    if dig.comboBox.currentIndex() == 9:
        sharpen(dig.image)
        
#FT

    if dig.comboBox.currentIndex() == 10:  
        dig.valueChannel = extractValueChannel(dig.image)
        dig.FT = fftpack.fft2(dig.valueChannel)
        v1=np.log(1+np.abs(dig.FT))
        plotoutput(v1)

#ShiftedFT        
    if dig.comboBox.currentIndex() == 11:
        ShiftedFT = fftpack.fftshift(dig.FT)
        v2=np.log(1+np.abs(ShiftedFT))
        plotoutput(v2)

#Log effect on Shifted FT      
    if dig.comboBox.currentIndex() == 12:
        dig.ShiftedFT = fftpack.fftshift(dig.FT) 
        v3=np.abs(dig.ShiftedFT)
        plotoutput(v3)

#LPF        
    if dig.comboBox.currentIndex() == 13:
        LPF = generateFilter(dig.ShiftedFT,0.5, 0.05, "LPF") 
        plotoutput(LPF)

#HPF        
    if dig.comboBox.currentIndex() == 14:    
        HPF = generateFilter(dig.ShiftedFT,0.025, 0.025, "HPF")
        plotoutput(HPF) 
         
def houghCircles():
    file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
    if file: # If the user gives a file
        dig.img= Image.open( file)
        
        dig.pixmap = QtGui.QPixmap(file) # Setup pixmap with the provided image
        dig.pixmap = dig.pixmap.scaled(dig.label_circles_input.width(), dig.label_circles_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        dig.label_circles_input.setPixmap(dig.pixmap) # Set the pixmap onto the label#dig.label_filters_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
           
        
        output_image1 = Image.new("RGB", dig.img.size)
        draw = ImageDraw.Draw(output_image1)
        for x, y in canny_edge_detector(dig.img):      
            draw.point((x, y), (255, 255, 255))
        output_image1.save("canny.png")
        
        dig.pixma = QtGui.QPixmap("canny.png") # Setup pixmap with the provided image
        dig.pixma = dig.pixma.scaled(dig.label_circles_hough.width(), dig.label_circles_hough.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        dig.label_circles_hough.setPixmap(dig.pixma) # Set the pixmap onto the label#dig.label_filters_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
        
        
        circle()
        # Output image with circles:
        output_image = Image.new("RGB", dig.img.size)
        output_image.paste(dig.img)
        draw_result = ImageDraw.Draw(output_image)
        for x, y, r in circles:
            draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))
        output_image.save("result.png")
        dig.pixm = QtGui.QPixmap("result.png") # Setup pixmap with the provided image
        dig.pixm = dig.pixm.scaled(dig.label_circles_output.width(), dig.label_circles_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        dig.label_circles_output.setPixmap(dig.pixm) # Set the pixmap onto the label#dig.label_filters_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
 ##############################


def histogram(img):
    height = img.shape[0]
    width = img.shape[1]
    
    hist = np.zeros((256))

    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i,j)
            hist[a] += 1
            
    return hist


def extractValueChannel(image):
    try:
        # Check if it has three channels or not 
        np.size(image, 2)
    except:
        return image
    hsvImage = col.rgb_to_hsv(image)
    return hsvImage[..., 2]

def generateFilter(image,w,h, filtType):
    if w > 0.5 or h > 0.5:
        print("w and h must be < 0.5")
        exit()
    m = np.size(image,0)
    n = np.size(image,1)
    LPF = np.zeros((m,n))
    HPF = np.ones((m,n))
    xi = np.round((0.5 - w/2) * m)
    xf = np.round((0.5 + w/2) * m)
    yi = np.round((0.5 - h/2) * n)
    yf = np.round((0.5 + h/2) * n)
    LPF[int(xi):int(xf),int(yi):int(yf)] = 1
    HPF[int(xi):int(xf),int(yi):int(yf)] = 0
    if filtType == "LPF":
        return LPF
    elif filtType == "HPF":
        return HPF
    else:
        print("Only Ideal LPF and HPF are supported")
        exit()        
def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)
def circle():
    # Find circles
    rmin = 20
    rmax = 25
    steps = 50
    threshold = 0.4
    
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
    
    acc = defaultdict(int)
    for x, y in canny_edge_detector(dig.img):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1
    global circles
    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            #print(v / steps, x, y, r)
            circles.append((x, y, r))   
             
#######################################################################
#Hough Lines
def houghLine(image):
    Ny = image.shape[0]
    Nx = image.shape[1]

    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90))
 #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            if image[y,x] > 0:
                for k in range(len(thetas)):
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + Maxdist,k] += 1
    return accumulator, thetas, rs


#prewitt and sobel operators 
########################################################################################
prewitt_h = np.array([[ -1 , 0 , 1 ] ,
                      [ -1 , 0 , 1 ] ,
                      [ -1 , 0 , 1 ] ])
prewitt_v = prewitt_h.transpose()


sobel_h = np.array([[ -1 , 0 , 1 ] ,
                    [ -2 , 0 , 2 ] ,
                    [ -1 , 0 , 1 ]])
sobel_v = sobel_h.transpose()
def sobel(img):
    image_sobel_h = signal.convolve2d( img, sobel_h ,'same')
    image_sobel_v = signal.convolve2d( img , sobel_v ,'same')
    phase = np.arctan2(image_sobel_h , image_sobel_v) * (180.0 / np.pi)
    phase = ((45 * np.round(phase / 45.0)) + 180) % 180;
    gradient = np.sqrt(image_sobel_h * image_sobel_h + image_sobel_v * image_sobel_v)
    plotoutput(gradient)


def prewit(img):
    image_prewit_h = signal.convolve2d( img , prewitt_h ,'same')
    image_prewit_v = signal.convolve2d( img , prewitt_v ,'same')
    gradient = np.sqrt(image_prewit_h * image_prewit_h + image_prewit_v * image_prewit_v)
    plotoutput(gradient)
       
#sharpen filter
#################################################################################################3
def sharpen(img):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im_sharpened = np.ones(img.shape)
    im_sharpened =signal.convolve2d(img, sharpen_kernel, mode='same')
    plotoutput(im_sharpened)

#laplacian filter
#################################################################################
def laplacian(img):
    lablacian_kernal = np.array([[ 0 , 1 , 0 ] ,
                                 [ 1 , -4 , 1 ] ,
                                 [ 0 , 1 , 0 ]])
    
    image_laplacian = signal.convolve2d( img , lablacian_kernal ,'same')
    plotoutput(image_laplacian)    
    
#log
#############################################################################
def log(img):
    lablacian_kernal = np.array([[ 0 , 1 , 0 ] ,
                                 [ 1 , -4 , 1 ] ,
                                 [ 0 , 1 , 0 ]])
    
    filtered_img_g7_std10 = signal.convolve2d(lablacian_kernal, gaussian_kernel(7,.3) ,'same')
    image_log = signal.convolve2d( img ,  filtered_img_g7_std10 ,'same')
    plotoutput(image_log)  
   
#dog
############################################################################
def dog(img):
    
    filtered_img_g7_std10 = signal.convolve2d(img, gaussian_kernel(7,1.4) ,'same')
    filtered_img_g5_std10 = signal.convolve2d(img, gaussian_kernel(5,1.4) ,'same')
    image_dog = filtered_img_g7_std10-filtered_img_g5_std10
    plotoutput(image_dog)  
 
#adding noise
###################################################################################
def addnoise(img):
    weight = 0.9
    noisy = img + weight * img.std() * np.random.random(img.shape)
    return noisy
 

#histogram
#####################################################################
def Histogram(img):
   row, col = img.shape # img is a grayscale image
   y = np.zeros((256), np.uint64)
   for i in range(0,row):
      for j in range(0,col):
         y[int(img[i,j])] += 1
   x = np.arange(0,256)
   plt.bar(x,y,color="gray",align="center")
   plt.show()
   return x,y

# create our cumulative function
def cdf(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)
   
def setimagehistogram():
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)") # Ask for file
    if fileName: # If the user gives a file
        
        dig.hisimage= mpimg.imread( fileName)
        dig.hisimage=rgb2gray(dig.hisimage)
        yourQImage=qimage2ndarray.array2qimage(dig.hisimage)
        gray=QtGui.QImage(yourQImage)
        pixmap  = QtGui.QPixmap.fromImage(gray)
        pixmap = pixmap.scaled(dig.label_histograms_input.width(), dig.label_histograms_input.height(), QtCore.Qt.KeepAspectRatio)
        dig.label_histograms_input.setPixmap( pixmap) # Set the pixmap onto the label
        dig.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter)   

        x,dig.y=Histogram(dig.hisimage)
        pw =pg.plot(x,dig.y) 


def HistogramEqualization():
    cs = cdf(dig.y)
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    # re-normalize the cdf on our image
    cs = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    img_new = cs[dig.his.flatten()]
    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, dig.his.shape)   
    
    yourQImage=qimage2ndarray.array2qimage(img_new)
    gray=QtGui.QImage(yourQImage)
    pixmap  = QtGui.QPixmap.fromImage(gray)
    pixmap = pixmap.scaled(dig.label_histograms_output.width(), dig.label_histograms_output.height(), QtCore.Qt.KeepAspectRatio)
    dig.label_histograms_output.setPixmap( pixmap) # Set the pixmap onto the label
    dig.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter) 
    
    z,w=Histogram(img_new)
    p=pg.plot(z,w)

#matching
def hist_match(source, template):
 

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    svalues, index, scounts = np.unique(source, return_inverse=True,return_counts=True)#
    tvalues, tcounts = np.unique(template, return_counts=True)
    sourcecdf = np.cumsum(scounts).astype(np.float64)#calculate comulative ndarray and cast the ndarray type 
    sourcecdf /= sourcecdf[-1]
    targetcdf = np.cumsum(tcounts).astype(np.float64)
    targetcdf /= targetcdf[-1]
    matched = np.interp(sourcecdf, targetcdf, tvalues)
    return matched[index].reshape(oldshape)        
def setimagetarget():
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)") # Ask for file
    if fileName: # If the user gives a file
        
        dig.targetimage= mpimg.imread( fileName)
        dig.targetimage=rgb2gray(dig.targetimage)
        
        yourQImage=qimage2ndarray.array2qimage(dig.hisimage)
        gray=QtGui.QImage(yourQImage)
        pixmap  = QtGui.QPixmap.fromImage(gray)
        pixmap = pixmap.scaled(dig.label_histograms_input.width(), dig.label_histograms_input.height(), QtCore.Qt.KeepAspectRatio)
        dig.label_histograms_input.setPixmap( pixmap) # Set the pixmap onto the label
        dig.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter)   
        x,y=Histogram(dig.targetimage)
        pg.plot(x,y,title='target histogram') 
def matching():
        matched=hist_match(dig.hisimage,dig.targetimage)
        yourQImage=qimage2ndarray.array2qimage(matched)
        gray=QtGui.QImage(yourQImage)
        pixmap  = QtGui.QPixmap.fromImage(gray)
        pixmap = pixmap.scaled(dig.label_histograms_output.width(), dig.label_histograms_output.height(), QtCore.Qt.KeepAspectRatio)
        dig.label_histograms_output.setPixmap( pixmap) # Set the pixmap onto the label
        dig.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)   
        x,y=Histogram(matched)
        pg.plot(x,y,title='matched histogram') 

#SNAKE
##################################################################################
def setimagesnake():
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            dig.snake= mpimg.imread( fileName)
            dig.snake=rgb2gray(dig.snake)
            #yourQImage=qimage2ndarray.array2qimage(dig.snake)
            #gray=QtGui.QImage(yourQImage)
            #pixmap  = QtGui.QPixmap.fromImage(gray)
            #pixmap = pixmap.scaled(dig.label_snake_image.width(), dig.label_snake_image.height(), QtCore.Qt.KeepAspectRatio)
            #dig.label_snake_image.setPixmap( pixmap) # Set the pixmap onto the label
            #dig.label_snake_image.setAlignment(QtCore.Qt.AlignCenter)   

def startsnake():

            t = np.arange(0, 2*np.pi,0.1)
            x = 437+350*np.cos(t)
            y = 493+400*np.sin(t)
#alpha =0.003
#beta  = 0.02
#gamma = 300
#iterations = 100

            alpha =float(dig.alpha.text())
            beta  =  float(dig.beta.text())
            gamma = float(dig.gamma.text())
            iterations = 100

# fx and fy are callable functions
            fx, fy = sn.create_external_edge_force_gradients_from_img(dig.snake, sigma=10 )
            snakes = sn.iterate_snake(x = x,y = y,a = alpha,b = beta,fx = fx,fy = fy,gamma = gamma,n_iters = iterations,return_all = True)
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            ax.imshow( dig.snake, cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0,dig.snake.shape[1])
            ax.set_ylim(dig.snake.shape[0],0)
            ax.plot(np.r_[x,x[0]], np.r_[y,y[0]], c=(0,1,0), lw=2)
            
            for i, snake in enumerate(snakes):
                if i % 10 == 0:
                    ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)

# Plot the last one a different color.

            ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)

            plt.show()
            
            
            yourQImage=qimage2ndarray.array2qimage(dig.snake)
            gray=QtGui.QImage(yourQImage)
            pixmap  = QtGui.QPixmap.fromImage(gray)
            pixmap = pixmap.scaled(dig.label_snake_image.width(), dig.label_snake_image.height(), QtCore.Qt.KeepAspectRatio)
            dig.label_snake_image.setPixmap( pixmap) # Set the pixmap onto the label
            dig.label_snake_image.setAlignment(QtCore.Qt.AlignCenter)        


    






 

###################################################################################        

    

app= QtWidgets.QApplication ([])
dig = uic.loadUi("mainwindow.ui")

dig.pushButton_filters_load.clicked.connect(setImage)
dig.Load_frequency_domain.clicked.connect(setImageFT)
dig.comboBox.activated[str].connect(setFilters)
dig.pushButton_lines_load.clicked.connect(HoughLines)
dig.pushButton_circles_load.clicked.connect(houghCircles)
dig.pushButton_histograms_load.clicked.connect(setimagehistogram)
dig.pushButton_histograms_load_target.clicked.connect(setimagetarget)
dig.pushButton_snake_load.clicked.connect(setimagesnake)
dig.pushButton_start.clicked.connect(startsnake)
dig.radioButton_2.toggled.connect(matching)
dig.Convert.clicked.connect(convetToGray)
dig.radioButton.toggled.connect(HistogramEqualization)


dig.show()
app.exec()



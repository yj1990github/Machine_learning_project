#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:41:16 2018

@author: fh

project 7

redo image processing for hallux
"""

from __future__ import print_function
import h5py    # HDF5 support
import numpy as np
#import imageio
import matplotlib.pyplot as plt
import cv2
import random

#from random import shuffle
#import six

import skimage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
import scipy.misc
from scipy import ndimage as ndi

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import transform as tf
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters.rank import otsu
from skimage.filters import  sobel
from skimage.feature import canny
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from skimage import feature
from skimage import measure
from skimage import filters
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max

#from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import random_walker

from skimage import data, util, filters, color
from skimage.exposure import rescale_intensity
from skimage import morphology
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage import exposure

#%% load data
fileName = "Dataset30HV.hdf5"
f = h5py.File(fileName,  "r")
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])
#OI= f['/entry/OI']
#OL= f['/entry/OL']
Xt= np.array(f['/entry/Xt'])   #30*800*800
Xlt=np.array( f['/entry/Xlt'])   #120*800*800
Xl=np.array(f['/entry/Xl'])    #120,
H= np.array(f['/entry/H'])     #30*100*100
PP= np.array(f['/entry/PP'])   #30 200 150
M1= np.array(f['/entry/M1'])    #(30, 400, 250)
M2= np.array(f['/entry/M2'])    #(30, 450, 200)
print(Xt.shape,Xlt.shape,Xl.shape,H.shape, PP.shape,M1.shape,M2.shape)
f.close()


#%%preprocessing

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#print('segmentation of bones')
#classes = ['hallux', 'proximal phalanx', 'metatarsal 1','metatarsal 2']
#num_classes = len(classes)
#samples_per_class = 2
#for y, cls in enumerate(classes):
#    idxs = np.flatnonzero(Xl == y+1)
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(Xlt[idx].astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()

temp=np.array([])
#temp1=np.array([])
#0-9 10-19 20-29
#fig 5 is bad
threshold1=[0.14, 0.05, 0.05, 0.25, 0.21, 0.04, 0.23, 0.23, 0.24, 0.25,\
           0.18, 0.16, 0.18, 0.14, 0.10, 0.39, 0.26, 0.18, 0.32, 0.005,\
           0.24, 0.14, 0.15, 0.13, 0.15, 0.1, 0.16, 0.03, 0.12, 0.1]
threshold2=[0.48, 0.29, 0.26, 0.43, 0.33, 0.08, 0.32, 0.47, 0.44, 0.44,\
           0.18, 0.16, 0.18, 0.14, 0.10, 0.39, 0.26, 0.18, 0.32, 0.005,\
           0.24, 0.14, 0.15, 0.13, 0.15, 0.1, 0.16, 0.03, 0.12, 0.1]
#for i in range(30):
#for i in range(9): #I manually find out the   
for i in [9]:
    print('Current number of image: '+repr(i))
    temp=denoise_tv_chambolle(Xt[i], weight=0.1, multichannel=True)
#    plt.hist(Xt[i].ravel())
#    plt.show()
#    plt.imshow(Xt[i])
#    plt.show()
#    plt.imshow(temp)
#    plt.show()
#    hist,bins = np.histogram(Xt[i].flatten(),256,[0,256]) 
#    cdf = hist.cumsum()
#    cdf_normalized = cdf * hist.max()/ cdf.max()
    p2, p98 = np.percentile(temp, (2, 98))
    temp2= exposure.rescale_intensity(temp, in_range=(p2, p98))
#    temp2 = exposure.equalize_hist(Xt[i])
    plt.imshow(temp2)
    plt.show()
#    plt.hist(temp2.ravel())
#    plt.show()
#    thresh_min=skimage.filters.threshold_minimum(temp2)
#    temp3=temp2.copy()
#    temp3[temp3<=thresh_min]==0
#    temp3[temp3>thresh_min]==1
#    temp2=cv2.equalizeHist(Xt[i])
#    Xt[i]=temp3
#    a=plt.hist(Xt[i].ravel())
#    plt.hist(Xt[i].ravel())
#    plt.show()

    
#    area=[]
#    for a in range(50):
#        thresh_min=a*0.01

#        temp3=temp2>thresh_min
##        print(a)
#        area.append(np.sum(temp3))
##        plt.imshow(temp3)
##        plt.show()
#    x_axis=[i for i in range(50)]
#    plt.plot(x_axis, area)
#    plt.show()
  
    
    #%%
    #function 
    def make_circle(a,b):#a=[x1,x2,x3] b=[y1,y2,y3]
    #given three point,return a circle
        assert len(a)==3
        assert len(b)==3
        
        x= a[0]+b[0]*(-1)**0.5
        y=a[1]+b[1]*(-1)**0.5
        z=a[2]+b[2]*(-1)**0.5
        w = z-x
        w /= y-x
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        circle=[-c.real, -c.imag, abs(c+x)] # x position, y position, r
        # https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
        return circle
    
    def p_axis(img):
        y0, x0 = np.nonzero(img)
        x = x0 - np.mean(x0)
        y = y0 - np.mean(y0)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
    #
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]
#        scale = 20
    #    plt.plot(x, y, 'k.')
#        plt.plot([x_v1*-scale*2+np.mean(x0), x_v1*scale*2+np.mean(x0)],
#         [y_v1*-scale*2+np.mean(y0), y_v1*scale*2+np.mean(y0)], color='red')
#        plt.plot([x_v2*-scale+np.mean(x0), x_v2*scale+np.mean(x0)],
#         [y_v2*-scale+np.mean(y0), y_v2*scale+np.mean(y0)], color='blue') 
        #The larger eigenvector is plotted in red and drawn twice as long as the smaller eigenvector in blue.
    #    plt.plot(x, y, 'k.')
    #    plt.axis('equal')
    #    plt.gca().invert_yaxis()  # Match the image system with origin at top left
#        plt.show()  
        vectors=[x_v1, x_v2, y_v1, y_v2]
        return vectors
    #https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    
    #%%
#    study the shape of bone and skin
    thresh_min=threshold1[i]
    temp3=temp2>thresh_min
    plt.imshow(temp3)
    plt.show()
    

   
    
    i_canny=canny(temp2)
    plt.imshow(i_canny)
#    plt.show()
#    i_fill = ndi.binary_fill_holes(i_canny)# bad
#    plt.imshow(i_fill)
    contours = measure.find_contours(temp3, 0.8)
    con_length=[i.shape[0] for i in contours]
    con_max=np.argmax(con_length)
#    for n, contour in enumerate(contours):
#        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.plot(contours[con_max][:, 1], contours[con_max][:, 0], linewidth=2)
#    plt.show()
    

    temp4=np.array(np.zeros(temp3.shape)).astype('int')
    temp4[temp3==False]=0
    temp4[temp3==True]=1
    M = measure.regionprops(temp4)   # find centroid
    centroid=M[0].centroid
    plt.plot(centroid[1],centroid[0],'ro')
    plt.show()
    
    D2_temp=contours[con_max]-np.transpose(np.array(centroid))#L2 Distance contour to centroid
    D2_temmp2=np.sum(D2_temp**2,axis=1)
    plt.plot(contours[con_max][:, 1],D2_temmp2)
    
    
    yhat = savgol_filter(D2_temmp2, 151, 4) # window size 51, polynomial order 3  smooth the curve
    plt.plot(contours[con_max][:, 1],yhat)
    a=argrelextrema(yhat, np.greater)
    b=argrelextrema(yhat, np.less)
    local_minimum0=0
    p0=[contours[con_max][b[0][local_minimum0],1],D2_temmp2[b[0][local_minimum0]]]
    local_maximum1=argrelextrema(yhat[a], np.greater)[0][0]
    p1=[contours[con_max][a[0][local_maximum1],1],D2_temmp2[a[0][local_maximum1]]]
#    plt.plot(p1,'ro')
#    local_maximum1=np.argmax(yhat[a]) # figure 9, =1
#    indx_temp=np.argmax(b[0]>a[0][local_maximum1])[0][]
#    local_minimum1=np.argmin(yhat[b][indx_temp:])+local_maximum1
    local_minimum1=argrelextrema(yhat[b][local_maximum1:], np.less)[0][0]+local_maximum1
    
    p2=[contours[con_max][b[0][local_minimum1],1],D2_temmp2[b[0][local_minimum1]]]
#    plt.plot(p2,'ro')
    local_maximum2=argrelextrema(yhat[a][local_minimum1:], np.greater)[0][0]+local_minimum1
    p3=[contours[con_max][a[0][local_maximum2],1],D2_temmp2[a[0][local_maximum2]]]
#    plt.plot(p3,'ro')
    local_minimum2=np.argwhere(b[0]>a[0][local_maximum2])[0]                   #have a problem
#    p4=[contours[con_max][local_minimum2,1],D2_temmp2[local_minimum2]]     # using argrelextrema
    p4=[contours[con_max][b[0][local_minimum2],1],D2_temmp2[b[0][local_minimum2]]]
    plt.plot([p0[0],p1[0],p2[0],p3[0],p4[0]],[p0[1],p1[1],p2[1],p3[1],p4[1]],'ro')     # succeed to plot the dots
#    indx_chosen=[local_maximum1,local_minimum1,local_maximum2]

    #effect threshold 10%
#    plt.plot(morphology.local_maxima())
    plt.show()
    
    

    
    plt.show()
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(Xt[i])
    indx_temp2=(b[0][local_minimum0],a[0][local_maximum1],b[0][local_minimum1],a[0][local_maximum2],b[0][local_minimum2])
    ax.plot(contours[con_max][indx_temp2, 1], contours[con_max][indx_temp2, 0],'ro')
    

# Now, loop through coord arrays, and create a circle at each x,y pair
    first_circle=(b[0][local_minimum0],a[0][local_maximum1],b[0][local_minimum1]) #p0 p1 p2
    second_circle=(b[0][local_minimum1],a[0][local_maximum2],b[0][local_minimum2]) #p2 p3 p4
    circle1=make_circle(contours[con_max][first_circle, 1], contours[con_max][first_circle, 0])
    
    ax.add_patch(plt.Circle((circle1[0], circle1[1]), circle1[2], color='r'))
    p0_c=[contours[con_max][b[0][local_minimum0], 1], contours[con_max][b[0][local_minimum0], 0]]
    p1_c=[contours[con_max][a[0][local_maximum1], 1], contours[con_max][a[0][local_maximum1], 0]]
    p2_c=[contours[con_max][b[0][local_minimum1], 1], contours[con_max][b[0][local_minimum1], 0]]
    p3_c=[contours[con_max][a[0][local_maximum2], 1], contours[con_max][a[0][local_maximum2], 0]]
    p4_c=[contours[con_max][b[0][local_minimum2], 1], contours[con_max][b[0][local_minimum2], 0]]
#    ax.add_patch(plt.Circle((400,400), 20, color='r'))
# Show the image
    plt.show()
    
    
    #%% 
    # find principal axis
    temp6=temp2.copy()
    for m in range(temp6.shape[0]):
        for n in range(temp6.shape[1]):
             if (m-circle1[1])**2+(n-circle1[0])**2>=circle1[2]**2:   #m--row---y  n--colomn--x
                 temp6[m,n]=0
    plt.imshow(temp6)
    plt.show()
#    temp6[temp6>filters.threshold_minimum(temp2)]=1
    temp6[temp6>threshold2[i]]=1
    temp6[temp6<=threshold2[i]]=0
    plt.imshow(temp6)
#    plt.show()
#    y0, x0 = np.nonzero(temp6)
#    x = x0 - np.mean(x0)
#    y = y0 - np.mean(y0)
#    coords = np.vstack([x, y])
#    cov = np.cov(coords)
#    evals, evecs = np.linalg.eig(cov)
##
#    sort_indices = np.argsort(evals)[::-1]
#    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
#    x_v2, y_v2 = evecs[:, sort_indices[1]]

    
    vectors_1=p_axis(temp6) #[x_v1, x_v2, y_v1, y_v2]
    y0, x0 = np.nonzero(temp6)
    [x_v1, x_v2, y_v1, y_v2]=vectors_1
    scale = 20
    plt.imshow(temp6)
##    plt.plot(x, y, 'k.')
    plt.plot([x_v1*-scale*2+np.mean(x0), x_v1*scale*2+np.mean(x0)],
     [y_v1*-scale*2+np.mean(y0), y_v1*scale*2+np.mean(y0)], color='red')
    plt.plot([x_v2*-scale+np.mean(x0), x_v2*scale+np.mean(x0)],
     [y_v2*-scale+np.mean(y0), y_v2*scale+np.mean(y0)], color='blue') 
#    #The larger eigenvector is plotted in red and drawn twice as long as the smaller eigenvector in blue.
##    plt.plot(x, y, 'k.')
##    plt.axis('equal')
##    plt.gca().invert_yaxis()  # Match the image system with origin at top left
    plt.show()  
    
#   #right line1: y-p2_c[1]= (vectors_1[2]/vectors_1[0])*(x-p2_c[0])
#   #left  line1: y-p0_c[1]= (vectors_1[2]/vectors_1[0])*(x-p0_c[0])
#   #up    line1: y-p1_c[1]= (vectors_1[3]/vectors_1[1])*(x-p1_c[0])
    leftbottom1=[0,p0_c[1] +(vectors_1[2]/vectors_1[0])*(0-p0_c[0])]
    rightip1_x=(p2_c[1]-p1_c[1]-(vectors_1[2]/vectors_1[0])*p2_c[0]+\
                (vectors_1[3]/vectors_1[1])*p1_c[0])/((vectors_1[3]/vectors_1[1])-(vectors_1[2]/vectors_1[0]))
    rightup1=[rightip1_x,p2_c[1]+ (vectors_1[2]/vectors_1[0])*(rightip1_x-p2_c[0])]
    leftup1_x=(p0_c[1]-p1_c[1]+(vectors_1[3]/vectors_1[1])*p1_c[0]-\
               (vectors_1[2]/vectors_1[0])*p0_c[0])/((vectors_1[3]/vectors_1[1])-(vectors_1[2]/vectors_1[0]))
    leftup1=[leftup1_x,p0_c[1]+ (vectors_1[2]/vectors_1[0])*(leftup1_x-p0_c[0])]
    x=[i for i in range(int(rightip1_x))]
    plt.plot(x,[int(z) for z in p2_c[1]+ (vectors_1[2]/vectors_1[0])*(x-p2_c[0])]) #right
    plt.plot(x,[int(z) for z in p0_c[1]+ (vectors_1[2]/vectors_1[0])*(x-p0_c[0])]) #left
    plt.plot(x,[int(z) for z in p1_c[1]+ (vectors_1[3]/vectors_1[1])*(x-p1_c[0])]) #up
    plt.plot(x,[int(z) for z in leftbottom1[1]+ (vectors_1[3]/vectors_1[1])*(x-np.array([0]))]) #bottom
#   #down  line1: y-p0_c[1] =(vectors_1[3]/vectors_1[1])*(0-p0_c[0])= (vectors_1[2]/vectors_1[0])*(x-0)?
    rightbottom1_x=(leftbottom1[1]-p2_c[1]+(vectors_1[2]/vectors_1[0])*(p2_c[0]))/(vectors_1[2]/vectors_1[0]-vectors_1[3]/vectors_1[1])
    rightbottom1=[rightbottom1_x,p2_c[1]+ (vectors_1[2]/vectors_1[0])*(rightbottom1_x-p2_c[0])]
#    plt.show()
    
#    plt.imshow(temp2)
    temp7=temp2.copy()

#    plt.plot(x,)
    plt.imshow(temp7)
#    indx_temp2=(b[0][local_minimum0],a[0][local_maximum1],b[0][local_minimum1],a[0][local_maximum2],b[0][local_minimum2])
    plt.plot(contours[con_max][indx_temp2, 1], contours[con_max][indx_temp2, 0],'ro')
    plt.plot([x_v1*-scale*2+np.mean(x0), x_v1*scale*2+np.mean(x0)],
     [y_v1*-scale*2+np.mean(y0), y_v1*scale*2+np.mean(y0)], color='red')
    plt.plot([x_v2*-scale+np.mean(x0), x_v2*scale+np.mean(x0)],
     [y_v2*-scale+np.mean(y0), y_v2*scale+np.mean(y0)], color='blue') 
#    plt.plot([leftbottom1[0],p0_c[0]-int(p0_c[1]*x_v1/y_v1)],[leftbottom1[1],0])
    plt.show()
    
    for m in range(temp7.shape[0]):
      for n in range(temp7.shape[1]):
         if m-p2_c[1]>= (vectors_1[2]/vectors_1[0])*(n-p2_c[0]):temp7[m,n]=0 #right
         elif m-p0_c[1]<= (vectors_1[2]/vectors_1[0])*(n-p0_c[0]):temp7[m,n]=0 #left
         elif m-p1_c[1]<= (vectors_1[3]/vectors_1[1])*(n-p1_c[0]):temp7[m,n]=0#up
         elif m-leftbottom1[1] >= (vectors_1[3]/vectors_1[1])*(n-np.array([0])):temp7[m,n]=0#bottom   #m--row---y  n--colomn--x
#                         temp7[m,n]=0
    plt.imshow(temp7)
    
    plt.plot([leftbottom1[0], rightup1[0], leftup1[0], rightbottom1[0]],[leftbottom1[1], rightup1[1], leftup1[1], rightbottom1[1]],'ro')
    plt.show()
    
    #%%
    #obtain tensity distribution and joints
    #hallux
    num_layers=int(((rightbottom1[0]-rightup1[0])**2+(rightbottom1[1]-rightup1[1])**2)**0.5)
    num_per=int(((rightbottom1[0]-leftbottom1[0])**2+(rightbottom1[1]-leftbottom1[1])**2)**0.5)
    
    #to righttop
    x_axis=[i for i in range(int(abs(rightbottom1[0]-rightup1[0])))]
    x_axis+=rightbottom1[0]
    y_axis=[int(z) for z in p2_c[1]+ (vectors_1[2]/vectors_1[0])*(x_axis-p2_c[0])]
    
    #to right bottom
    x_axis2=[i for i in range(int(abs(rightbottom1[0]-leftbottom1[0])))]
    x_axis2=rightbottom1[0]-x_axis2
    y_axis2=[int(z) for z in leftbottom1[1]+ (vectors_1[3]/vectors_1[1])*(x-np.array([0]))]
    
    temp7=temp7>threshold2[i]
    temp7=ndi.binary_fill_holes(temp7)
    y_prime=[]    
    for num in range(len(x_axis)):
        temp=0
        x_temp=x_axis[num]-[i for i in range(int(abs(rightbottom1[0]-leftbottom1[0])))]
        y_temp=[int(z) for z in y_axis[num]+ (vectors_1[3]/vectors_1[1])*(x_temp-x_axis[num])]
        
        for num2 in range(len(x_temp)):
            temp+=temp7[int(y_temp[num2]),int(x_temp[num2])]
        y_prime.append(temp)
    print(len(y_prime))
    plt.plot([i for i in range(len(y_prime))],y_prime)
    plt.show()
    plt.plot([i for i in range(len(y_prime)-1)],\
              (np.array(y_prime[:-1])-np.array(y_prime[1:]).tolist()))#find derivative
    plt.show()
    max_deriv=np.argmax(np.array(y_prime[:-1])-np.array(y_prime[1:]))
    joint_first0=[x_axis[max_deriv],\
                 y_axis[max_deriv]]
    joint_first=[x_axis[max_deriv]-int(0.5*abs(rightbottom1[0]-leftbottom1[0])),\
                 y_axis[max_deriv]-int(0.5*(vectors_1[3]/vectors_1[1])*abs(rightbottom1[0]-leftbottom1[0]))]
#    x,[int(z) for z in p1_c[1]+ (vectors_1[3]/vectors_1[1])*(x-p1_c[0])]
    plt.plot(joint_first[0],joint_first[1],'ro')
#    plt.plot(joint_first0[1],joint_first0[0],'ro')
    plt.plot(joint_first0[0],joint_first0[1],'ro')
    plt.imshow(temp7)

    plt.plot(x_axis,y_axis)
    plt.show()

    for m in range(temp7.shape[0]):
      for n in range(temp7.shape[1]):
         if m-joint_first[1] >= (vectors_1[3]/vectors_1[1])*(n-joint_first[0]):temp7[m,n]=0#bottom   #m--row---y  n--colomn--x
    plt.imshow(temp7)

       
    contours7 = measure.find_contours(temp7, 0.8)
    con_length7=[i.shape[0] for i in contours7]
    con_max7=np.argmax(con_length7)
    plt.plot(contours7[con_max7][:, 1],contours7[con_max7][:, 0])
    middle_point=[int(0.5*(contours7[con_max7][(np.argmin(contours7[con_max7][:, 1])), 1]+\
                           contours7[con_max7][(np.argmax(contours7[con_max7][:, 1])), 1])),
                  int(0.5*(contours7[con_max7][(np.argmin(contours7[con_max7][:, 0])), 0]+\
                           contours7[con_max7][(np.argmax(contours7[con_max7][:, 0])), 0]))]
    #fill in the contour
    x_array=np.array(contours7[con_max7][:, 1]).astype('int')
    y_array=np.array(contours7[con_max7][:, 0]).astype('int')
    

#%%
# try horitonzal swweep
    temp8=np.zeros_like(temp2)
    y_max0=y_array[np.argmax(y_array)]
    y_min0=y_array[np.argmin(y_array)]
    for yy in range(y_max0-y_min0+1):
        indx=np.argwhere(y_array==y_min0+yy)
        if indx.shape[0]==0:
            print('none:  '+repr(yy))
            pass
        else:
        
#        try:
            temp=[ x_array[i] for i in indx]
            x_min1=temp[np.argmin(temp)]
            x_max1=temp[np.argmax(temp)]
            temp8[y_min0+yy,x_min1[0]:x_max1[0]+1]=1
            print('good:   '+repr(x_min1)+ ' '+repr(x_max1))
            plt.imshow(temp8)
            plt.show()            
#        except: 
#            pass
    plt.imshow(temp8)
    plt.show()
            
    
    
    #%% 
    #contour filling   so bad, time consuming
#    area=1
#    temp=0
#    temp_x=[middle_point[0]]
#    temp_y=[middle_point[1]]
    
#    temp9=np.zeros_like(temp2)
#    for xx in range(len(x_array)):
#        temp9[x_array[xx],y_array[xx]]=1
#              
#    plt.plot(middle_point[0],middle_point[1],'ro')
#    plt.show() 
#    
#    temp8=np.zeros_like(temp2)
#    temp8[middle_point[0],middle_point[1]]=1
#    def detect(x,y,contour_x,contour_y):
##        assert len(x)==len(y)
#        
#        for i in range(len(contour_x)):
#            if (contour_x[i]-x)**2+(contour_y[i]-y)**2==0:
##                print('found')
#                return False
#        else:return True
#                
#    while temp!=area:
##    for xx in range(1):
#        area=temp
##        left=temp_x[np.argmin(temp_x)]
##        right=temp_x[np.argmax(temp_x)]
##        up=temp_x[np.argmin(temp_y)]
##        bottom=temp_x[np.argmin(temp_y)]
#        [temp_y,temp_x]=np.nonzero(np.array(temp8))
#        for o in range(len(temp_x)):
#            if detect(temp_x[o]+1,temp_y[o],x_array,y_array) and temp8[temp_x[o]+1,temp_y[o]]!=1:
#                temp8[temp_x[o]+1,temp_y[o]]=1
#            if detect(temp_x[o]-1,temp_y[o],x_array,y_array) and temp8[temp_x[o]-1,temp_y[o]]!=1:
#                temp8[temp_x[o]-1,temp_y[o]]=1
#            if detect(temp_x[o],temp_y[o]+1,x_array,y_array)and temp8[temp_x[o],temp_y[o]+1]!=1:
#                temp8[temp_x[o],temp_y[o]+1]=1
#            if detect(temp_x[o],temp_y[o]-1,x_array,y_array) and temp8[temp_x[o],temp_y[o]-1]!=1:
#                temp8[temp_x[o],temp_y[o]-1]=1 
##            if detect(temp_x[o]+1,temp_y[o]+1,x_array,y_array) and temp8[temp_x[o]+1,temp_y[o]+1]!=1:
##                temp8[temp_x[o]+1,temp_y[o]+1]=1
##            if detect(temp_x[o]-1,temp_y[o]+1,x_array,y_array) and temp8[temp_x[o]-1,temp_y[o]+1]!=1:
##                temp8[temp_x[o]-1,temp_y[o]+1]=1  
##            if detect(temp_x[o]-1,temp_y[o]-1,x_array,y_array) and temp8[temp_x[o]-1,temp_y[o]-1]!=1:
##                temp8[temp_x[o]-1,temp_y[o]-1]=1
##            if detect(temp_x[o]+1,temp_y[o]-1,x_array,y_array)and temp8[temp_x[o]+1,temp_y[o]-1]!=1:
##                temp8[temp_x[o]+1,temp_y[o]-1]=1
#        temp=np.sum(temp8)
#        print(temp)
#        plt.imshow(temp8)
#        plt.plot(contours7[con_max7][:, 1],contours7[con_max7][:, 0])
#        plt.show()
#    
#    plt.imshow(temp8)
#    plt.show()
    
  #%%
#study the shape of bone  
    temp5=temp2>threshold2[i]
    plt.imshow(temp5)
#    intensity_a=[i for i in range(800)]   # results are not good
#    intensity_b=(800-np.sum(temp4,axis=0)).tolist()
#    plt.plot(intensity_a, intensity_b)
    
    contours_5 = measure.find_contours(temp5, 0.8)
    con_length_5=[i.shape[0] for i in contours_5]
    con_max_5=np.argmax(con_length_5)
#    for n, contour in enumerate(contours):
#        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
#    plt.plot(contours_5[con_max_5][:, 1], contours_5[con_max_5][:, 0], linewidth=2)
#    plt.show()
    
    D2_temp3=contours_5[con_max_5]-np.transpose(np.array(centroid))#L2 Distance contour to centroid
    D2_temp4=np.sum(D2_temp3**2,axis=1)
#    plt.plot(contours_5[con_max_5][:, 1],D2_temp4)

#    temp3=temp2>threshold_otsu(temp2)-0.35
#    block_size = 35
#    local_thresh = threshold_local(temp2, block_size, offset=-0.4)
#    temp3=temp2>local_thresh
    
#    selem = disk(15)
#    local_otsu = rank.otsu(temp2, selem)
#    temp3=temp2>local_otsu
#    plt.show()


    
#    plt.imshow(temp3)
#    plt.show()
#    a=[i for i in range(800)]
#    b=(800-np.sum(temp3,axis=0)).tolist()
#    plt.plot(a, b)
#    plt.show()
    
    try:
#        temp=np.vstack((temp,np.array([Xlt0[4*i][:400,:600]])))
#        temp1=np.vstack((temp,np.array([Xt0[i][:400,:600]])))
        temp=np.vstack((temp,np.array([Xlt[4*i]])))
#        temp1=np.vstack((temp,np.array([Xt0[i])))
    except:
#        temp=np.array([Xlt0[4*i][:400,:600]])
#        temp1=np.array([Xt0[i][:400,:600]])
        temp=np.array([Xlt[4*i]])
#        temp1=np.array([Xt0[i]])
#    plt.imshow(Xlt[4*i])
#    plt.show()
Xlt1=temp  #obtain feacture one
print(Xlt.shape)
#Xt=temp  #obtain feacture one
#print(Xt.shape)

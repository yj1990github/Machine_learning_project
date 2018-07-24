#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:17:50 2018

@author: fh
Buid a dataset with h5 file format using 30-training data 
"""

import h5py    # HDF5 support
import numpy as np
#import imageio
import matplotlib.pyplot as plt
import cv2
import six





#%%image preprocessing
    #rescale the image to have row pixel number=1000
    #find the dimension of each labels
    #integrate them into features with same dimension
#img=[]#original images
#img_l=[] #original labels ,manually instance segmentation

#img_r=[]#rescaled images
#H_patch=[]
#PP_patch=[]
#M1_patch=[]
#M2_patch=[]

#row-y axis, col-x axis
#shape: y-axis, x-axis
#resize: x-axis  y-axis, 

#select train1 as a base line
img1=cv2.imread('Xtraining_dataset/train'+repr(1)+'.jpg')
img1_g=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img=[img1_g]
label1=cv2.imread('Ytraining_dataset/Label0_'+repr(0+1)+'.png') # read original images
label1_g=cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY) #rbg to gray
#img_l=[label1_g]

#row_d=1000#desired image size
#scal=img1_g.shape[0]/row_d#scaling factor
#size_d=(int(scal*img1_g.shape[1]),row_d)
#img1_r=cv2.resize(img1_g, size_d,interpolation = cv2.INTER_AREA)
#img_r+=[img1_r]

H_label1=label1_g.copy()
H_label1[H_label1!=1]=0
H_xmax1=np.argwhere(np.sum(H_label1>0,axis=0))[-1][0]
H_xmin1=np.argwhere(np.sum(H_label1>0,axis=0))[0][0]
H_ymax1=np.argwhere(np.sum(H_label1>0,axis=1))[-1][0]
H_ymin1=np.argwhere(np.sum(H_label1>0,axis=1))[0][0]
#    plt.imshow(H_label[H_ymin:H_ymax,H_xmin:H_xmax])
#    plt.show()
H_label_t1=H_label1[H_ymin1:H_ymax1,H_xmin1:H_xmax1]

#desired patch size 100*100
if H_label_t1.shape[0]>=H_label_t1.shape[1]:
    H_label_r1=cv2.resize(H_label_t1,(int(100*H_label_t1.shape[1]/H_label_t1.shape[0]),100))
    #cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming
else:
    H_label_r1=cv2.resize(H_label_t1,(100,int(100*H_label_t1.shape[0]/H_label_t1.shape[1])) )
H_label_r1[H_label_r1!=0]=1
H_label_d0=np.zeros((100,100))
H_label_d0[50-int(H_label_r1.shape[0]/2):50-int(H_label_r1.shape[0]/2)+H_label_r1.shape[0],\
           50-int(H_label_r1.shape[1]/2):50-int(H_label_r1.shape[1]/2)+H_label_r1.shape[1]]=H_label_r1
#plt.imshow(H_label_d1)
#plt.show()
H_patch=np.array([H_label_d0])

scal1=H_label_t1.shape[0]/100#scaling factor
img1_r0=cv2.resize(img1_g, (int(img1_g.shape[1]/scal1),int(img1_g.shape[0]/scal1)) )
label1_g0=label1_g.copy()
label1_g0[label1_g0!=1]=0
label1_g0=cv2.resize(label1_g0, (int(label1_g0.shape[1]/scal1),int(label1_g0.shape[0]/scal1)) )
label1_g0[label1_g0!=0]=1
label2_g0=label1_g.copy()
label2_g0[label2_g0!=2]=0
label2_g0=cv2.resize(label2_g0, (int(label2_g0.shape[1]/scal1),int(label2_g0.shape[0]/scal1)) )
label2_g0[label2_g0!=0]=1
label3_g0=label1_g.copy()
label3_g0[label3_g0!=3]=0
label3_g0=cv2.resize(label3_g0, (int(label3_g0.shape[1]/scal1),int(label3_g0.shape[0]/scal1)) )
label3_g0[label3_g0!=0]=1
label4_g0=label1_g.copy()
label4_g0[label4_g0!=4]=0
label4_g0=cv2.resize(label4_g0, (int(label4_g0.shape[1]/scal1),int(label4_g0.shape[0]/scal1)) )
label4_g0[label4_g0!=0]=1
#desired image size 800*800
if img1_r0.shape[0]>800:
    img1_r0=img1_r0[:800,:]
    label1_g0=label1_g0[:800,:]
    label2_g0=label2_g0[:800,:]
    label3_g0=label3_g0[:800,:]
    label4_g0=label4_g0[:800,:]
else:
    img1_r0=np.vstack((img1_r0,np.zeros((800-img1_r0.shape[0],img1_r0.shape[1]))))
    label1_g0=np.vstack((label1_g0,np.zeros((800-label1_g0.shape[0],label1_g0.shape[1]))))
    label2_g0=np.vstack((label2_g0,np.zeros((800-label2_g0.shape[0],label2_g0.shape[1]))))
    label3_g0=np.vstack((label3_g0,np.zeros((800-label3_g0.shape[0],label3_g0.shape[1]))))
    label4_g0=np.vstack((label4_g0,np.zeros((800-label4_g0.shape[0],label4_g0.shape[1]))))
if img1_r0.shape[1]>800:
    img1_r0=img1_r0[:,int(img1_r0.shape[1]/2)-400:int(img1_r0.shape[1]/2)+400]
    label1_g0=label1_g0[:,int(label1_g0.shape[1]/2)-400:int(label1_g0.shape[1]/2)+400]
    label2_g0=label2_g0[:,int(label2_g0.shape[1]/2)-400:int(label2_g0.shape[1]/2)+400]
    label3_g0=label3_g0[:,int(label3_g0.shape[1]/2)-400:int(label3_g0.shape[1]/2)+400]
    label4_g0=label4_g0[:,int(label4_g0.shape[1]/2)-400:int(label4_g0.shape[1]/2)+400]
else:
    img1_r0=np.hstack((img1_r0,np.zeros((img1_r0.shape[0],800-img1_r0.shape[1]))))
    label1_g0=np.hstack((label1_g0,np.zeros((label1_g0.shape[0],800-label1_g0.shape[1]))))
    label2_g0=np.hstack((label2_g0,np.zeros((label2_g0.shape[0],800-label2_g0.shape[1]))))
    label3_g0=np.hstack((label3_g0,np.zeros((label3_g0.shape[0],800-label3_g0.shape[1]))))
    label4_g0=np.hstack((label4_g0,np.zeros((label4_g0.shape[0],800-label4_g0.shape[1]))))
    
img_r=np.array([img1_r0])#########################################################################
img_l=np.array([label1_g0])
img_l=np.vstack((img_l,[label2_g0]))
img_l=np.vstack((img_l,[label3_g0]))
img_l=np.vstack((img_l,[label4_g0]))


plt.imshow(img1_r0)
plt.show() 
#plt.imshow(label1_g0)
#plt.show() 
#plt.imshow(img1_r0)
#plt.show() 
#plt.imshow(H_label_d0)
#plt.show()   
for i in range(3):
    label_temp=label1_g.copy()
    label_temp[label_temp!=i+2]=0
#    plt.imshow(label_temp)
#    plt.show()
    xmax=np.argwhere(np.sum(label_temp>0,axis=0))[-1][0]
    xmin=np.argwhere(np.sum(label_temp>0,axis=0))[0][0]
    ymax=np.argwhere(np.sum(label_temp>0,axis=1))[-1][0]
    ymin=np.argwhere(np.sum(label_temp>0,axis=1))[0][0]
    label_temp2=label_temp[ymin:ymax,xmin:xmax]
#    plt.imshow(label_temp2)
#    plt.show()
    label_temp3=cv2.resize(label_temp2,(int(label_temp2.shape[1]/scal1),int(label_temp2.shape[0]/scal1)) )
    label_temp3[label_temp3!=0]=1
#    plt.imshow(label_temp3)
#    plt.show()
    
    if i==0:   
            label_temp4=np.zeros((200,150))
            label_temp4[100-int(label_temp3.shape[0]/2):100-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
               75-int(label_temp3.shape[1]/2):75-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
            PP_patch=np.array([label_temp4])
    elif i==1:
            label_temp4=np.zeros((400,250))
            label_temp4[200-int(label_temp3.shape[0]/2):200-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
               125-int(label_temp3.shape[1]/2):125-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
            M1_patch=np.array([label_temp4])
    else:
            label_temp4=np.zeros((450,200))
            label_temp4[225-int(label_temp3.shape[0]/2):225-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
               100-int(label_temp3.shape[1]/2):100-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
            M2_patch=np.array([label_temp4])
#    plt.imshow(label_temp4)
#    plt.show()

#size of base patches
        #(100, 69) (149, 87) (292, 106) (333, 63)
#desired dimension of each ROIs
        #(100, 100) (200, 150) (350, 150) (400, 100)
#print(H_patch[0].shape,PP_patch[0].shape,M1_patch[0].shape,M2_patch[0].shape)

#H_label_r=cv2.resize(H_label_t, (int(H_label_t.shape[1]/scal),int(H_label_t.shape[0]/scal)),interpolation = cv2.INTER_AREA)
#H_label_r2=H_label_r.copy()
#H_label_r2[H_label_r2!=0]=1
#H_label_d1=np.vstack((H_label_r2,np.zeros((200-H_label_r2.shape[0],H_label_r2.shape[1]))))
#H_label_d2=np.hstack((H_label_d1,np.zeros((H_label_d1.shape[0],200-H_label_d1.shape[1]))))

#plt.imshow(cv2.cvtColor(cv2.imread('Label_3.png'), cv2.COLOR_BGR2GRAY))

for j in range(30):
    if j==0:
        continue
    image=cv2.imread('Xtraining_dataset/train'+repr(j+1)+'.jpg') # read original images
    img_g=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #rbg to gray
#    img+=[img_g]
    label=cv2.imread('Ytraining_dataset/Label0_'+repr(j+1)+'.png') # read original images
    label_g=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) #rbg to gray
#    img_l+=[label_g]
#    plt.imshow(label_g)
#    plt.show()

#   resize all figures to match H_patch
    H_label=label_g.copy()
    H_label[H_label!=1]=0
    H_xmax=np.argwhere(np.sum(H_label>0,axis=0))[-1][0]
    H_xmin=np.argwhere(np.sum(H_label>0,axis=0))[0][0]
    H_ymax=np.argwhere(np.sum(H_label>0,axis=1))[-1][0]
    H_ymin=np.argwhere(np.sum(H_label>0,axis=1))[0][0]
    #    plt.imshow(H_label[H_ymin:H_ymax,H_xmin:H_xmax])
    #    plt.show()
    H_label_t=H_label[H_ymin:H_ymax,H_xmin:H_xmax]
    
    #desired patch size 100*100
    if H_label_t.shape[0]>=H_label_t.shape[1]:
        scal2=H_label_t.shape[0]/100#scaling factor
        H_label_r=cv2.resize(H_label_t,(int(100*H_label_t.shape[1]/H_label_t.shape[0]),100) )
    else:
        scal2=H_label_t.shape[1]/100#scaling factor
        H_label_r=cv2.resize(H_label_t,(100,int(100*H_label_t.shape[0]/H_label_t.shape[1])) )
    H_label_r[H_label_r!=0]=1
    H_label_d1=np.zeros((100,100))
    H_label_d1[50-int(H_label_r.shape[0]/2):50-int(H_label_r.shape[0]/2)+H_label_r.shape[0],\
               50-int(H_label_r.shape[1]/2):50-int(H_label_r.shape[1]/2)+H_label_r.shape[1]]=H_label_r
    #plt.imshow(H_label_d1)
    #plt.show()
    H_patch=np.vstack((H_patch,[H_label_d1]))
    
    
    img1_r=cv2.resize(img_g, (int(img_g.shape[1]/scal2),int(img_g.shape[0]/scal2)) )
    label1_g1=label_g.copy()
    label1_g1[label1_g1!=1]=0
    label1_g1=cv2.resize(label1_g1, (int(label1_g1.shape[1]/scal2),int(label1_g1.shape[0]/scal2)) )
    label1_g1[label1_g1!=0]=1
    label2_g1=label_g.copy()
    label2_g1[label2_g1!=2]=0
    label2_g1=cv2.resize(label2_g1, (int(label2_g1.shape[1]/scal2),int(label2_g1.shape[0]/scal2)) )
    label2_g1[label2_g1!=0]=1
    label3_g1=label_g.copy()
    label3_g1[label3_g1!=3]=0
    label3_g1=cv2.resize(label3_g1, (int(label3_g1.shape[1]/scal2),int(label3_g1.shape[0]/scal2)) )
    label3_g1[label3_g1!=0]=1
    label4_g1=label_g.copy()
    label4_g1[label4_g1!=4]=0
    label4_g1=cv2.resize(label4_g1, (int(label4_g1.shape[1]/scal2),int(label4_g1.shape[0]/scal2)) )
    label4_g1[label4_g1!=0]=1
#    print(img1_r.shape,label_g1.shape)
        #desired image size 800*800
    if img1_r.shape[0]>800:
        img1_r=img1_r[:800,:]
        label1_g1=label1_g1[:800,:]
        label2_g1=label2_g1[:800,:]
        label3_g1=label3_g1[:800,:]
        label4_g1=label4_g1[:800,:]
    else:
        img1_r=np.vstack((img1_r,np.zeros((800-img1_r.shape[0],img1_r.shape[1]))))
        label1_g1=np.vstack((label1_g1,np.zeros((800-label1_g1.shape[0],label1_g1.shape[1]))))
        label2_g1=np.vstack((label2_g1,np.zeros((800-label2_g1.shape[0],label2_g1.shape[1]))))
        label3_g1=np.vstack((label3_g1,np.zeros((800-label3_g1.shape[0],label3_g1.shape[1]))))
        label4_g1=np.vstack((label4_g1,np.zeros((800-label4_g1.shape[0],label4_g1.shape[1]))))
    if img1_r.shape[1]>800:
        img1_r=img1_r[:,int(img1_r.shape[1]/2)-400:int(img1_r.shape[1]/2)+400]
        label1_g1=label1_g1[:,int(label1_g1.shape[1]/2)-400:int(label1_g1.shape[1]/2)+400]
        label2_g1=label2_g1[:,int(label2_g1.shape[1]/2)-400:int(label2_g1.shape[1]/2)+400]
        label3_g1=label3_g1[:,int(label3_g1.shape[1]/2)-400:int(label3_g1.shape[1]/2)+400]
        label4_g1=label4_g1[:,int(label4_g1.shape[1]/2)-400:int(label4_g1.shape[1]/2)+400]
    else:
        img1_r=np.hstack((img1_r,np.zeros((img1_r.shape[0],800-img1_r.shape[1]))))
        label1_g1=np.hstack((label1_g1,np.zeros((label1_g1.shape[0],800-label1_g1.shape[1]))))
        label2_g1=np.hstack((label2_g1,np.zeros((label2_g1.shape[0],800-label2_g1.shape[1]))))
        label3_g1=np.hstack((label3_g1,np.zeros((label3_g1.shape[0],800-label3_g1.shape[1]))))
        label4_g1=np.hstack((label4_g1,np.zeros((label4_g1.shape[0],800-label4_g1.shape[1]))))
        
    img_r=np.vstack((img_r,[img1_r]))#########################################################################
    img_l=np.vstack((img_l,[label1_g1]))
    img_l=np.vstack((img_l,[label2_g1]))
    img_l=np.vstack((img_l,[label3_g1]))
    img_l=np.vstack((img_l,[label4_g1]))

    
    
#    print(img_r[j].shape)
#    plt.imshow(img1_r)
#    plt.show()
#    plt.imshow(img_g)
#    plt.show() 
#    plt.imshow(label_g1)
#    plt.show() 
#    plt.imshow(img1_r)
#    plt.show() 
#    plt.imshow(H_label_d1)
#    plt.show()   
#    plt.imshow(img_l[i])
#    plt.show()
#    print(i+1,image.shape,label.shape)
#    print(i+1,img[i].shape,img_l[i].shape)
#    shape1+=[img[i].shape]
#    shape2+=[img_l[i].shape]
#plt.imshow(f)
#plt.show()
#
#for j in range(30):
#    idx=[w for w, v in enumerate(shape2) if v==shape1[j]]
#    print(repr(j+1)+ '+'+repr(idx[0]+1))

    for k in range(3):
            label_temp=label_g.copy()
            label_temp[label_temp!=k+2]=0
            #    plt.imshow(label_temp)
            #    plt.show()
            xmax=np.argwhere(np.sum(label_temp>0,axis=0))[-1][0]
            xmin=np.argwhere(np.sum(label_temp>0,axis=0))[0][0]
            ymax=np.argwhere(np.sum(label_temp>0,axis=1))[-1][0]
            ymin=np.argwhere(np.sum(label_temp>0,axis=1))[0][0]
            label_temp2=label_temp[ymin:ymax,xmin:xmax]
            #    plt.imshow(label_temp2)
            #    plt.show()
            label_temp3=cv2.resize(label_temp2,(int(label_temp2.shape[1]/scal2),int(label_temp2.shape[0]/scal2)) )
            label_temp3[label_temp3!=0]=1
            #    plt.imshow(label_temp3)
            #    plt.show()
            #        
            if (k==0):   
                    label_temp4=np.zeros((200,150))
                    label_temp4[100-int(label_temp3.shape[0]/2):100-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
                       75-int(label_temp3.shape[1]/2):75-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
                    PP_patch=np.vstack((PP_patch,[label_temp4]))
            elif (k==1):
                    label_temp4=np.zeros((400,250))
                    label_temp4[200-int(label_temp3.shape[0]/2):200-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
                       125-int(label_temp3.shape[1]/2):125-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
                    M1_patch=np.vstack((M1_patch,[label_temp4]))
            else:
                    label_temp4=np.zeros((450,200))
                    label_temp4[225-int(label_temp3.shape[0]/2):225-int(label_temp3.shape[0]/2)+label_temp3.shape[0],\
                       100-int(label_temp3.shape[1]/2):100-int(label_temp3.shape[1]/2)+label_temp3.shape[1]]=label_temp3
                    M2_patch=np.vstack((M2_patch,[label_temp4]))
#            plt.imshow(label_temp4)
#            plt.show()

Xt0=img_r
Xt1=img_l
Xl0=np.array([1,2,3,4])
for l in range(29):
    Xl0=np.hstack((Xl0,[1,2,3,4]))
####
#img: preprocessed x ray images (List(No., jpgs))
#img_l: instance segmentation of img (List(No., pngs))
#img_r: rescaled imgae
#0:background, 1:hallux(H), 2:proximal phalanx(PP), 3:first metatarsal(M1), 4:second metarsal(M2)  
# H_patch  PP_patch M1_patch  M2_patch
#label
    
###


#%%
# ###################### write hdf5 file        need to uncomment this section
#    
fileName = u"Dataset30HV.hdf5"
timestamp = u"2018-6-18Thu11:10:00"

## load data from two column format
#data = [17.92608,1037]
#mr_arr = data[0]
#i00_arr = np.asarray(data[1],'int32')

# create the HDF5 NeXus file
f = h5py.File(fileName, "w")
# point to the default data to be plotted
f.attrs[u'default']          = u'entry'
# give the HDF5 root some more attributes
f.attrs[u'file_name']        = fileName
f.attrs[u'file_time']        = timestamp
f.attrs[u'creator']          = u'data_processing.py'
f.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
f.attrs[u'h5py_version']     = six.u(h5py.version.version)

# create the first group
entry = f.create_group(u'entry')
#nxentry.attrs[u'NX_class'] = u'NXentry'
entry.attrs[u'default'] = u'x-ray'
#entry.attrs[u'Original_images'] = u'OI'
#entry.create_dataset(u'OI', data=img)
#entry.attrs[u'Original_labels'] = u'OL'
#entry.create_dataset(u'OL', data=img_l)
#entry.attrs[u'Rescaled_images'] = u'RI'
#entry.create_dataset(u'RI', data=img_r)
#entry.create_dataset(u'RL', data=label_r)
entry.create_dataset(u'Xt', data=Xt0)
entry.create_dataset(u'Xlt', data=Xt1)
entry.create_dataset(u'Xl', data=Xl0)
#entry.attrs[u'H_patch'] = u'H'
entry.create_dataset(u'H', data=H_patch)   #100*100
#entry.attrs[u'PP_patch'] = u'PP'
entry.create_dataset(u'PP', data=PP_patch)   #200*150
#entry.attrs[u'M1_patch'] = u'M1'
entry.create_dataset(u'M1', data=M1_patch)  #400*250
#entry.attrs[u'M2_patch'] = u'M2'
entry.create_dataset(u'M2', data=M2_patch)  #450*200
## create the NXentry group
#nxdata = nxentry.create_group(u'mr_scan')
#nxdata.attrs[u'NX_class'] = u'NXdata'
#nxdata.attrs[u'signal'] = u'I00'      # Y axis of default plot
#nxdata.attrs[u'axes'] = u'mr'         # X axis of default plot
#nxdata.attrs[u'mr_indices'] = [0,]   # use "mr" as the first dimension of I00
#
## X axis data
#ds = nxdata.create_dataset(u'mr', data=mr_arr)
#ds.attrs[u'units'] = u'degrees'
#ds.attrs[u'long_name'] = u'USAXS mr (degrees)'    # suggested X axis plot label
#
## Y axis data
#ds = nxdata.create_dataset(u'I00', data=i00_arr)
#ds.attrs[u'units'] = u'counts'
#ds.attrs[u'long_name'] = u'USAXS I00 (counts)'    # suggested Y axis plot label
#
f.close()   # be CERTAIN to close the file

#%%
#######################
#read the file                                   need to uncomment this section
#
fileName = "Dataset30HV.hdf5"
f = h5py.File(fileName,  "r")
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])
#OI= f['/entry/OI']
#OL= f['/entry/OL']
Xt= f['/entry/Xt']
Xlt= f['/entry/Xlt']
Xl=f['/entry/Xl']
H= f['/entry/H']
PP= f['/entry/PP']
M1= f['/entry/M1']
M2= f['/entry/M2']
print(Xt.shape,Xlt.shape,Xl.shape,H.shape, PP.shape,M1.shape,M2.shape)
#i00 = f['/entry/mr_scan/I00']
#print("%s\t%s\t%s" % ("#", "mr", "I00"))
#for i in range(len(mr)):
#    print("%d\t%g\t%d" % (i, mr[i], i00[i]))
f.close()
    
    
#%%
#    
#    
###save new images to jpg files
###    
#model0=np.zeros_like(img1)
#model=model0[:100,:100,:]
#for m in range(img_r.shape[0]):
#    img1=model.copy()
#    img10=cv2.resize(img_r[m],(100,100) )
#    img1[:,:,0]=img10
#    img1[:,:,1]=img10
#    img1[:,:,2]=img10
##    img1=np.vstack((np.array([img01]),np.array([img01])))
##    img1=np.vstack((np.array([img1]),np.array([img01])))
##    print(img1.shape)
##    plt.imshow(img1)
##    plt.show()
#    img2=img_l[m]
#    img21=img2.copy()
#    img21[img21!=1]=0
#    img210=cv2.resize(img21,(100,100) )
#    img210[img210!=0]=1
#    
##    img22=img2.copy()
##    img22[img22!=1]=0 
##    img22[img22!=0]=1 
##    img23=img2.copy()
##    img23[img23!=1]=0
##    img23[img23!=0]=1 
##    img24=img2.copy()
##    img24[img24!=1]=0 
##    img24[img24!=0]=1
#    
##    plt.imshow(img21)
##    plt.show()
##    plt.imshow(img22)
##    plt.show()
##    plt.imshow(img23)
##    plt.show()
##    plt.imshow(img24)
##    plt.show()
#    img021=model.copy()
#    img021[:,:,0]=img210
#    img021[:,:,1]=img210
#    img021[:,:,2]=img210
#    
##    img022=model.copy()
##    img022[:,:,0]=img22
##    img022[:,:,1]=img22
##    img022[:,:,2]=img22
##    img023=model.copy()
##    img023[:,:,0]=img23
##    img023[:,:,1]=img23
##    img023[:,:,2]=img23
##    img024=model.copy()
##    img024[:,:,0]=img24
##    img024[:,:,1]=img24
##    img024[:,:,2]=img24
#    
##    img021=np.vstack((np.array([img21]),np.array([img21])))
##    img021=np.vstack((np.array([img021]),np.array([img21])))
##    print(img021.shape)
##    img022=np.vstack((np.array([img22]),np.array([img22])))
##    img022=np.vstack((np.array([img022]),np.array([img22])))
##    print(img021.shape) 
##    img023=np.vstack((np.array([img23]),np.array([img23])))
##    img023=np.vstack((np.array([img023]),np.array([img23])))
##    print(img021.shape)
##    img024=np.vstack((np.array([img24]),np.array([img24])))
##    img024=np.vstack((np.array([img024]),np.array([img24])))
##    print(img021.shape,img022.shape,img023.shape,img024.shape)
#    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/images/image'+repr(m+1)+'.jpg',img1)
##    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps/label'+repr(m+1)+'1.jpg',img021)
##    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps/label'+repr(m+1)+'2.jpg',img022)
##    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps/label'+repr(m+1)+'3.jpg',img023)
##    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps/label'+repr(m+1)+'4.jpg',img024)
#    cv2.imwrite('/Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps/image'+repr(m+1)+'.png',img021)
#    
##    /Users/fh/Desktop/boston/2018_summer/hallux_valgus_project/image/Dataset30HV/annotations/trimaps
#    
#    
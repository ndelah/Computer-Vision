# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:22:40 2020

@author: delah
"""
#%% IMPORTS
import cv2
import os
import numpy as np
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1')
print(os.getcwd())

## Checklist:
# Put the gaussian and bilateral filter images side by side with unaltered image
# Grab object in HSV color space
# Add color to sobel eedge detector
# Hough transofrm and 
# Object detection


#%% 2.2 GAUSSIAN that increase by widening the kernel

# Import video
cap = cv2.VideoCapture('in\\gaussian.mp4')
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\gaussian.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height))

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
count = 1
blur_factor = 101

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1
    if ret==True:
        ksize = (blur_factor,blur_factor)
        frame = cv2.GaussianBlur(frame,ksize=ksize,sigmaX=0)
        
        #Adds subtitles
        cv2.putText(frame,text="Gaussian Filter: ksize = {}".format(ksize), org=(200,50),fontFace=font,thickness=1,fontScale=1,color=(0,0,255))
        cv2.putText(frame,text="Resets when Ksize == (101,101)", org=(200,450),fontFace=font,thickness=1,fontScale=1,color=(0,0,255))

        # write the Blurred frame
        out.write(frame)
        cv2.imshow('frame',frame)  
        
        if count % 2 == 0 and count <= 100:
            blur_factor -= 2
        elif count > 100:
                blur_factor = 1            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

#%% 2.2 BILATERAL that increase by widening the kernel
# Parameters
cap = cv2.VideoCapture('in\\gaussian.mp4')
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out\\bilateral.mp4',cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height))
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
i = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1
    if ret==True:
        frame = cv2.bilateralFilter(frame,d=i,sigmaColor=50,sigmaSpace=50)
        cv2.putText(frame,text="Bilateral Filter: d={}, sigmaColor = 50, sigmaSpace=50".format(i), org=(200,50),fontFace=font,thickness=1,fontScale=1,color=(0,0,0))
        # write the Blurred frame
        out.write(frame)
        cv2.imshow('frame',frame)
        if count % 10 == 0 and i != 15:
            i += 1
        elif i == 15:
            i = 15
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
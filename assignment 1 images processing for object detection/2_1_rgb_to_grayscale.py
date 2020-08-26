# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:30:06 2020

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


#%% Part 2:
# 2.1 Switch between color and gray scale

# Import video
cap = cv2.VideoCapture('in\\bw.mp4')
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\gray.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height),isColor=1)

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
count = 1
colored = True

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1   
    if ret==True:
        if colored == True:          
            cv2.putText(frame,"Changes between RGB and Grayscale every 100 frames",org=(200,700),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            cv2.imshow('frame',frame)
            out.write(frame)
            if count % 20 == 0:
                colored = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break          
        elif colored == False:
            cv2.putText(frame,"Changes between RGB and Grayscale every 100 frames",org=(200,700),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            cv2.imshow('frame',frame)
            out.write(frame)
            if count % 20 == 0:
                colored = True
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        else:
            break        
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:23:17 2020

@author: delah
"""
#%% IMPORTS
import cv2
import os
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1')
print(os.getcwd())

## Checklist:
# Put the gaussian and bilateral filter images side by side with unaltered image
# Grab object in HSV color space
# Add color to sobel eedge detector
# Hough transofrm and 
# Object detection

#%% 2.3 Object with white foreground and black backround : Thresholding
# Improve grabbing with fill holdes and undetectede edges by using bninary morphological opreations. Put improvements in a different color

# Import video
cap = cv2.VideoCapture('in\\vid2.mp4')
fps = int(round(cap.get(5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\threshold2.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                
        # Using the inverse and subtracting fram1 - frame2 looks nicer
        res1, frame1 = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
        res2, frame2 = cv2.threshold(closing,150,255,cv2.THRESH_BINARY_INV)
        difference = cv2.subtract(frame1, frame2)
        
        ret, mask = cv2.threshold(difference, 0, 255,cv2.THRESH_BINARY) 
        thresh2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        thresh2[mask != 0] = [100, 0, 255]
        out.write(thresh2)        
        cv2.imshow('frame',thresh2)
    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


#%% 2.3 Object with white foreground and black backround : Thresholding
# Improve grabbing with fill holdes and undetectede edges by using bninary morphological opreations. Put improvements in a different color

# Import video
cap = cv2.VideoCapture('in\\vid3.mp4')
fps = int(round(cap.get(5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\threshold.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_HSV2)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                
        # Using the inverse and subtracting fram1 - frame2 looks nicer
        res1, frame1 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY_INV)
        res2, frame2 = cv2.threshold(closing,90,255,cv2.THRESH_BINARY_INV)
        difference = cv2.subtract(frame1, frame2)
        
        ret, mask = cv2.threshold(difference, 0, 255,cv2.THRESH_BINARY) 
        thresh2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        thresh2[mask != 0] = [100, 0, 255]
        out.write(thresh2)        
        cv2.imshow('frame',thresh2)
    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
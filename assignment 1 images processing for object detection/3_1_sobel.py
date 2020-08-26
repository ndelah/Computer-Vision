# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:24:29 2020

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


#%% Function definition

"""
Parameters:
    frame = cap.read() type frame
    style = "horizontal","vertical" "both" Determines which style of Sobel filter to apply
    ksize = must be uneven. -1 for a Scharr filter. Usually looks best with 3
    
Description:
Blurs, converts video to grayscale and the applies the Sobel Filter. 
The 
"""

def  sobel_function(frame,style,ksize,delta):
        frame = cv2.blur(frame,ksize=(5,5))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(frame, ddepth=-1, dx=1, dy=0,ksize=ksize,scale=1,delta=delta)
        grad_y = cv2.Sobel(frame, ddepth=-1, dx=0, dy=1,ksize=ksize,scale=1,delta=delta)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        difference_x = cv2.subtract(abs_grad_x, frame)
        difference_y = cv2.subtract(abs_grad_y, frame)

        ret, mask_x = cv2.threshold(difference_x, 0, 255,cv2.THRESH_BINARY) 
        ret, mask_y = cv2.threshold(difference_y, 0, 255,cv2.THRESH_BINARY) 
        
        thresh_y = cv2.cvtColor(mask_y, cv2.COLOR_GRAY2BGR)
        thresh_x = cv2.cvtColor(mask_x, cv2.COLOR_GRAY2BGR)

        if style == "vertical":
            grad = thresh_x
            grad[mask_x != 0] = [0,0,255]
        elif style == "horizontal":
            grad = thresh_y
            grad[mask_y != 0] = [255,0,0]

        elif style == "both":
            grad = cv2.addWeighted(thresh_x, 0.5, thresh_y, 0.5, 0)
            grad[mask_x != 0] = [206,113,255]
            grad[mask_y != 0] = [254,205,1]

        return grad
#%% Part 3: Object detection
# 3.1 Sobel horizontal and vertical edge detection

# Import video
cap = cv2.VideoCapture('in\\vid3.mp4')
fps=cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\both.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))
font = cv2.FONT_HERSHEY_SIMPLEX

# Parameters
delta = 0
count = 1
increase = True
style = "both"
ksize = 3

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1
    if ret:
        if increase == True :
            frame = sobel_function(frame,style=style,ksize=ksize,delta=delta)
            cv2.putText(frame,"Sobel filter: style = {}, ksize = {}, delta = {}".format(style,ksize,delta) ,org=(200,100),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            cv2.imshow('frame',frame)
            out.write(frame)
            delta += 1
            
            if count % 60 == 0:
                increase = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        elif increase == False:
            frame = sobel_function(frame,style="both",ksize=3,delta=delta)
            cv2.putText(frame,"Sobel filter: style = {}, ksize = {}, delta = {}".format(style,ksize,delta) ,org=(200,100),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            cv2.imshow('frame',frame)
            out.write(frame)
            delta -= 1
            if count % 60 == 0:
                increase = True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
 
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
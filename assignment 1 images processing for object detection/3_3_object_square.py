# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:26:16 2020

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


#%% 3_3 Object Square
# Track object in scene with a rectangle (2 seconds)
# gray scale with intensity of values proportional to likelihood of object of interest at that location (5 seconds)

# Import video
cap = cv2.VideoCapture('in\\wallet.mp4')
#cap = cv2.VideoCapture(1)
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\object_detection2.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))


template = cv2.imread('in\\wallet.jpg',0)
#template = cv2.resize(template,(width-100,height-100))
w, h = template.shape[::-1]

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1
    if ret==True:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            ogres = cv2.matchTemplate(gray_frame,template,cv2.TM_SQDIFF)
            res = cv2.resize(ogres,(width,height))
            res = cv2.convertScaleAbs(res)
            res = cv2.addWeighted(src1=res,alpha=0.2,src2=gray_frame,beta=0.8,gamma=0.8)
            #res = np.uint8(cv2.cvtColor(res,cv2.COLOR_GRAY2BGR) *255)
            #Adds subtitles
            cv2.putText(frame,text="Object Detection", org=(200,700),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
            # write the Blurred frame
            cv2.imshow('frame',res)  
            out.write(res)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                       
    else:
        break
    
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

#%% 3_3 Object Square
# Track object in scene with a rectangle (2 seconds)
# gray scale with intensity of values proportional to likelihood of object of interest at that location (5 seconds)

# Import video
cap = cv2.VideoCapture('in\\rolling.mp4')
#cap = cv2.VideoCapture(1)
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\object_detectionpingpong.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))


template = cv2.imread('in\\rolling.jpg',0)
#template = cv2.resize(template,(width-100,height-100))
w, h = template.shape[::-1]

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1
    if ret==True:
        if count < 60:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            res = cv2.matchTemplate(gray_frame,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.55
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
                
    
            #Adds subtitles
            cv2.putText(frame,text="Object Detection", org=(200,700),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
    
            # write the Blurred frame
            out.write(frame)
            cv2.imshow('frame',frame)  
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif count >= 60:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            ogres = cv2.matchTemplate(gray_frame,template,cv2.TM_SQDIFF)
            res = cv2.resize(ogres,(width,height))
            res = cv2.convertScaleAbs(res)
            res = cv2.addWeighted(src1=res,alpha=0.2,src2=gray_frame,beta=0.8,gamma=0.2)
            #res = np.uint8(cv2.cvtColor(res,cv2.COLOR_GRAY2BGR) *255)
            #Adds subtitles
            cv2.putText(frame,text="Object Detection", org=(200,700),fontFace=font,thickness=1,fontScale=1,color=(255,255,255))
            
            # write the Blurred frame
            cv2.imshow('frame',res)  
            out.write(res)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                       
    else:
        break
    
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
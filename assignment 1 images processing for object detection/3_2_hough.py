# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:25:19 2020

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

#%% test

# Parameters
cap = cv2.VideoCapture("in\\vid2.mp4")
#cap = cv2.VideoCapture(1)

fps = int(round(cap.get(5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3,640)
cap.set(4,480)
nameOutput = "out\\houghall.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height))
font = cv2.FONT_HERSHEY_DUPLEX

dp=0.01
mindist = 10
minradius= 5
maxradius = 50
param1=50
param2=20

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:        
        canny = cv2.Canny(frame, 100,200)
        circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT,dp=dp,minDist=mindist,
                                    param1=param1,param2=param2,minRadius=minradius,maxRadius=maxradius)
        
        
        cqircles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)        
        
        cv2.putText(frame,"Hough filter dp = {}, mindist = {}, minradius = {}".format(dp,mindist,minradius),org=(200,100),fontFace=font,thickness=1,fontScale=1,color=(255,0,0))
        cv2.putText(frame,"maxradius = {}, param1 = {}, param2={}  ".format(maxradius,param1,param2),org=(200,125),fontFace=font,thickness=1,fontScale=1,color=(255,0,0))

        cv2.imshow('detected circles',frame)
        out.write(frame)
        
        if mindist < 120:
            mindist += 1
        if dp <1 :
            dp += 0.01
            dp = round(dp,2)
        if minradius < 30:
            minradius += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
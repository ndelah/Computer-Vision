# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:27:40 2020

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



#%% Carte Blanche
# Parameters
cap = cv2.VideoCapture('in\\broll.mp4')
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out\\vaporwave.mp4',cv2.VideoWriter_fourcc(*'DIVX'),fps,(width,height))
font = cv2.FONT_HERSHEY_SIMPLEX

rows,cols = cap.shape


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
            # write the flipped frame
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = np.stack([frame,frame,frame],axis=2)
            
        
        M = np.float32([[1,0,100],[0,1,50]])
        dst = cv2.warpAffine(frame,M,(cols,rows))
                
        #frame = cv2.applyColorMap(frame, colormap=cv2.COLORMAP_PINK)
        cv2.imshow('frame',dst)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
        else:
            break
        
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()





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
        
        #Pourquoi on est obligé de changé le mask_x en thresh???
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
            #grad[mask_x == 0] = [150,251,255]
            #grad[mask_y == 0] = [161,255,5]

        return grad

#%% Part 3: Object detection
# 3.1 Sobel horizontal and vertical edge detection

# Warp affine 
# dst = cv2.warpAffine(frame,M,(cols,rows))   
        
    
# Import video
cap = cv2.VideoCapture('in\\vid3.mp4')
fps=cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\carteblanche.mp4"
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

#%%



def main():

    img = vaporize()

    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow("pic", img)

    while cv2.getWindowProperty("pic", cv2.WND_PROP_VISIBLE):
        key_code = cv2.waitKey(100)

        if key_code == ESCAPE_KEY:
            break
        elif key_code != -1:
            import time
            start = time.time()
            img = vaporize()
            cv2.imshow("pic", img)
            end = time.time()
            logger.info("Vaporizing and rendering took: %f seconds" % (end-start,))
    cv2.destroyAllWindows()
    sys.exit()



#%% Add eleemnts

import logging
import os
import random as rd
from functools import partial

import cv2

logger = logging.getLogger("elements")


def add_elements(img):
    min_elements = 2
    max_elements = 4

    base_dir = "elements/black/"

    all_files = os.listdir(base_dir)
    rd.shuffle(all_files)

    # randomize number of elements added
    num_elements = rd.randint(min_elements, max_elements)
    # create a set to prevent element repetition
    added_counter = 0

    logger.info("Adding %d elements" % (num_elements, ))

    for file_name in map(partial(os.path.join, base_dir), all_files):
        if added_counter == num_elements:
            return

        success = add_single_element(img, file_name)
        if success:
            added_counter += 1


def add_single_element(img, file_name):
    imh, imw, imd = img.shape
    element = cv2.imread(file_name, -1)
    if element is None:
        logger.warning("Could not read file %s" % (file_name,))
        return False

    original_height, original_width, original_depth = element.shape
    # adjust size if too big
    if original_height > imh * .5 or original_width > imw * .5:
        element = cv2.resize(element, (int(.5 * original_width), int(.5 * original_height)))

        resized_height, resized_width, _ = element.shape
        # refuse to use this image, if this failed
        if resized_height > imh or resized_width > imw:
            logger.warning("Element %s too big, moving on" % (file_name,))
            return False
        # get x coord and y coord on the image
        from_x_pos = rd.randint(1, imw - resized_width - 1)
        from_y_pos = rd.randint(1, imh - resized_height - 1)
        # make alpha channel
        alpha_s = element[:, :, 2] / 255.0
        alpha_1 = 1.0 - alpha_s
        for c in range(0, 3):
            to_y_pos = from_y_pos + resized_height
            to_x_pos = from_x_pos + resized_width

            with_alpha_s = alpha_s * element[:, :, c]
            with_alpha_1 = alpha_1 * img[from_y_pos:to_y_pos, from_x_pos:to_x_pos, c]

            img[from_y_pos:to_y_pos, from_x_pos:to_x_pos, c] = with_alpha_s + with_alpha_1
    return True




#%%
import cv2
from Vapormaster.vaporwave import vaporize

# Load the cascade
face_cascade = cv2.CascadeClassifier('in\\eyes.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(1)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()


#%%
import sys
import logging
import os 
import cv2
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1\\Vapormaster')
print(os.getcwd())

from vaporwave import vaporize



ESCAPE_KEY = 27

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main():

    img = vaporize()

    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow("pic", img)

    while cv2.getWindowProperty("pic", cv2.WND_PROP_VISIBLE):
        key_code = cv2.waitKey(100)

        if key_code == ESCAPE_KEY:
            break
        elif key_code != -1:
            import time
            start = time.time()
            img = vaporize()
            cv2.imshow("pic", img)
            end = time.time()
            logger.info("Vaporizing and rendering took: %f seconds" % (end-start,))
    cv2.destroyAllWindows()
    sys.exit()
# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#%%
import numpy as np
import cv2
import os
import random as rd
import logging
from functools import partial
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1')
print(os.getcwd())

cap = cv2.VideoCapture("in\\tomask.mp4")
img2 = cv2.imread("in\\background.jpg")

fps = int(round(cap.get(5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\vaporwave2.mp4"
out = cv2.VideoWriter(nameOutput,cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height))

# Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
count = 0


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        rows = frame.shape[0]
        cols = frame.shape[1]
        channels = frame.shape[2]
        
        roi = img2[0:rows, 0:cols ]
        
        add_elements(frame)

        
        #frame = vaporize(frame)
    
        #MASKING part of me 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Using the inverse
        res1, mask = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        
        img2_fg = cv2.bitwise_and(frame,frame,mask = mask)
        
    
        dst = cv2.add(img1_bg,img2_fg)
        frame[0:rows, 0:cols ] = dst
        
        #thresh2[mask != 0] = [100, 0, 255]   
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        out.write(frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#%%

import numpy as np
import cv2
import os
import random as rd
import logging
from functools import partial

import cv2

logger = logging.getLogger("elements")


def add_elements(img):
    min_elements = 2
    max_elements = 4

    base_dir = "in\\elements\\black"

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

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
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


#%% Face recognition
import cv2
import os
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1')
print(os.getcwd())
# Load the cascade
face_cascade = cv2.CascadeClassifier('in\\haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture("in\\gaussian.mp4")
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
fps = cap.get(5)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out\\facerecognition.mp4',cv2.VideoWriter_fourcc(*'MP4V'),fps,(width,height))
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:       
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        cv2.imshow('frame', frame)
        out.write(frame)
        count += 1
        
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    else:
        break
# Release the VideoCapture object
cap.release()
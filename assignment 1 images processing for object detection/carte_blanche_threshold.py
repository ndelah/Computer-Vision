#%% IMPORTS
import cv2
import os
os.chdir('C:\\Users\\delah\\Google Drive\\University\\KULeuven Master of Artificial Intelligence\\Second Semester\\2 Computer Vision\\2020\\assignment\\1')
print(os.getcwd())

#%%
# Import video
cap = cv2.VideoCapture('in\\tomask.mp4')
fps = int(round(cap.get(5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nameOutput = "out\\thresholdtest.mp4"
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
        res1, mask = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

        frame[mask == 0] = [100, 0, 255]
        #thresh2[mask != 0] = [100, 0, 255]
        out.write(frame)        
        cv2.imshow('frame',frame)
    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

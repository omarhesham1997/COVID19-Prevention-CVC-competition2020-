# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
from operator import itemgetter


#capture video as an example
cap = cv2.VideoCapture('v1.mp4')


#Saving The output video
cap.set(cv2.CAP_PROP_POS_FRAMES, 1335)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('project.avi',fourcc, 20, (800,225))

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# improve detection accuracy by decreasing width
min_Distance=70
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()
    
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    		padding=(8, 8), scale=1.05)
    
#sorting according to distance x axis
    rects=sorted(rects, key=itemgetter(0))
    

    #Drawing Red Circles on people of Unsafe Position and
    # green circles on people in safe position
    for i in range(len(rects)-1):
        if( i==0):
            cv2.circle(orig,(int(rects[i][0]+rects[i][2]/2),int(rects[i][1]+rects[i][3]/2)),5,(0, 255, 0),2)   
        distance=rects[i+1][0]-rects[i][0]
        print(distance)
        if distance< min_Distance:
           cv2.circle(orig,(int(rects[i+1][0]+rects[i+1][2]/2),int(rects[i+1][1]+rects[i+1][3]/2)),5,(0, 0, 255),2)
        else:
           cv2.circle(orig,(int(rects[i+1][0]+rects[i+1][2]/2),int(rects[i+1][1]+rects[i+1][3]/2)),5,(0, 255, 0),2) 

        
    frame = cv2.resize(orig,(400,225), interpolation = cv2.INTER_CUBIC)    
    vis = np.concatenate((image, frame), axis=1)
    out.write(vis)

    # show the output images
    cv2.imshow("output", orig)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
cap.release()
out.release()
cv2.destroyAllWindows()

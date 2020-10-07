'''
References:

1. Changing Colorspaces, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces

2. Image Thresholding, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding

3. Contour Features, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features

4. Smoothing Images, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    # Take each frame
    ret,frame = cap.read()

    if ret==True:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # define range of blue color in HSV
        lower_blue_hsv = np.array([90,80,10])
        upper_blue_hsv = np.array([120,255,255])
        
        # define range of blue color in RGB
        lower_blue_rgb = np.array([10,30,80])
        upper_blue_rgb = np.array([60,150,255])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        # define range of the green color in HSV
        # lower_green_hsv = np.array([50,80,50])
        # upper_green_hsv = np.array([70,170,255])

        # define range of the green color in RGB
        # lower_green_rgb = np.array([60,160,60])
        # upper_green_rgb = np.array([170,255,170])

        # Threshold the HSV image to get only blue colors
        mask_hsv = cv2.inRange(hsv,lower_blue_hsv,upper_blue_hsv)

        # Threshold the BGR image to get only blue colors
        mask_rgb = cv2.inRange(rgb,lower_blue_rgb,upper_blue_rgb)

        # Threshold the HSV image to get only green colors
        # mask_hsv = cv2.inRange(hsv,lower_green_hsv,upper_green_hsv)

        # Threshold the RGB image to get only green colors
        # mask_rgb = cv2.inRange(rgb,lower_green_rgb,upper_green_rgb)

        # Filter masked image using median blur
        median_hsv = cv2.medianBlur(mask_hsv,13)
        median_rgb = cv2.medianBlur(mask_rgb,13)

        # Bitwise-AND mask and original image
        # res_hsv = cv2.bitwise_and(frame,frame,mask=median_hsv)
        # res_rgb = cv2.bitwise_and(frame,frame,mask=median_rgb)

        contours_hsv,_ = cv2.findContours(median_hsv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_rgb,_ = cv2.findContours(median_rgb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_hsv) > 0:
            cnt = max(contours_hsv, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if (area > 600):
                # Draw coordinates of centroid
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                frame = cv2.circle(frame,(int(cx),int(cy)),2,(255,0,0),2)
                cv2.putText(frame,str(cx)+", "+str(cy),(cx+5,cy-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

                # Draw bounding box
                x,y,w,h = cv2.boundingRect(cnt) 
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,"HSV",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2) 

        if len(contours_rgb) > 0:
            cnt = max(contours_rgb, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if (area > 600):
                # Draw coordinates of centroid
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                frame = cv2.circle(frame,(int(cx),int(cy)),2,(0,255,0),2)
                cv2.putText(frame,str(cx)+", "+str(cy),(cx+5,cy-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                # Draw bounding box
                x,y,w,h = cv2.boundingRect(cnt) 
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,"RGB",(x+w-50,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2) 

        cv2.imshow('frame',frame)
        cv2.imshow('HSV mask',median_hsv)
        cv2.imshow('RGB mask',median_rgb)
        # cv2.imshow('HSV res',res_hsv)
        # cv2.imshow('RGB res',res_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
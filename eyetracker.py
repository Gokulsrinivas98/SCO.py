# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pynput.mouse import Button, Controller
import os
import sys
from tkinter import *

flag=0
mouse = Controller()
t1=0
t2=0
xnew=676
ynew=385
time1=0
time2=0
timed=0
eye_cascade=cv2.CascadeClassifier('parojosG.xml')
direction="center"
cnt=0
roi_gray=cv2.imread('randompic2.jpg')
kernel=np.ones((3,3),np.uint8)
resize=cv2.imread('randompic2.jpg')
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",	help="path to input video file")
args = vars(ap.parse_args())
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2


# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	#frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray=cv2.equalizeHist(gray)



	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
		 	#then increment the total number of blinks
        		if COUNTER >= EYE_AR_CONSEC_FRAMES:
				                TOTAL += 1




				                if(TOTAL % 2==0):
                                                     flag=0
                                                     t2=time.time()
				                else:
                                                    flag=1
                                                    t1=time.time()
				                if (flag==0):
				                	time1=t2-t1
				                elif(flag==1):
				                	time2=t1-t2
				                timed=time1+time2
				                if(timed<=3):
				                	print("Hello")
				                	mouse.click(Button.left, 1)

				# reset the eye frame counter
        		COUNTER = 0
        		#print(t2-t1)
		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	eyes=eye_cascade.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in eyes:
		w=int(w/2)
		roi_gray=gray[y:y+h,x:x+w]
		ret,roi_gray = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		roi_gray=cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
		roi_gray = cv2.medianBlur(roi_gray,5)
		roi_gray = cv2.GaussianBlur(roi_gray,(5,5),0)


		h1=int(h/2)+1
		w1=int(w/4)+1
		w2=int(w1*3)-17
		#roi_gray[0:h1,0:w1]=0
		#roi_gray[h1:h,w2:w]=0
		pt1=roi_gray[h1,w1]
		pt2=roi_gray[h1,w2]
		if(pt1==0):
			direction="right"
			if (flag==0):
				xnew=xnew+5
			elif(flag==1):
				ynew=ynew+5
		elif(pt2==0):
			direction="left"
			if (flag==0):
				xnew=xnew-5
			elif(flag==1):
				ynew=ynew-5
		else:
			direction="center"
	roi_gray=cv2.morphologyEx(roi_gray,cv2.MORPH_OPEN,kernel)

	cv2.imshow('eye',roi_gray)
	#cv2.imshow('eye1',roi_gray1)
	if(cnt==2):
		#print(direction)
		mouse.position = (xnew, ynew)
		cnt=0
	cnt+=1

	# if the `q` key was pressed, break from the loop


	if key == 27:
		break

	# if the number of blinks is 2, break from the loop
	#if TOTAL == 2:
	#	import eyeCurrent2
	#	TOTAL=0

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


import cv2
import sys
import math
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import struct
import pickle
import os
import serial
import logging
from flask import Flask, render_template
from flask_ask import Ask, statement, question, audio
import thread
from twilio.rest import Client
import pafy

client = Client("AC763a630070363e34f3b9689c702e3b2a", "b7c265fb97698401816bbd2712e49eab")

inFrame = True
personDetected = True
threadOn = False
SLEEP = False
trackFail = False

TeamNames = ['Arvand', 'Erick', 'Renooka']



arduino = serial.Serial('/dev/ttyACM0', 9600)
arduino.write(struct.pack('>B',0))

app = Flask(__name__)
ask = Ask(app, "/")

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)
logging.getLogger("flask_ask").setLevel(logging.DEBUG)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

#TeamNames=['Arvand', 'Erick', 'Renooka']
followName=''

if not args.get("video", False):
	video = cv2.VideoCapture(1)

    # Read video
else:
    	video = cv2.VideoCapture(args["video"])
 
    # Exit if video not opened.
if not video.isOpened():
	print "Could not open video"
        sys.exit()
 
    # Read first frame.
ok, frame = video.read()
if not ok:
	print 'Cannot read video file'
	sys.exit()

############################################### Face recognition function
def find_face():
	global followName
	n=0
	name=''
	bbox=[0,0,0,0]
	while n<10:
		_,FRAME = video.read()
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		FRAME = imutils.resize(FRAME, width=600)
		(h, w) = FRAME.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(FRAME, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = FRAME[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				#print name.shape


				# draw the bounding box of the face along with the
				# associated probability
				y = startY - 10 if startY - 10 > 10 else startY + 10
				#print "{}: {:.2f}%".format(name, proba * 100)
				bbox = startX, startY, endX-startX, endY-startY
				
				if name == followName:
					return bbox, name

		n = n+1
	return bbox, name

##################################################

######################################Tracking algorithm
def tracking():
	arduino.write(struct.pack('>B',0))
	global inFrame
	global personDetected
	global lost_frames
	global trackFail
	trackFail = False
	tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    	tracker_type = tracker_types[2]
 
    	if int(minor_ver) < 3:
        	tracker = cv2.Tracker_create(tracker_type)
   	else:
		if tracker_type == 'BOOSTING':
		    	tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
		    	tracker40 = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
		    	tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
		    	tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
		    	tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
		    	tracker = cv2.TrackerGOTURN_create()
		if tracker_type == 'MOSSE':
		    	tracker = cv2.TrackerMOSSE_create()
		if tracker_type == "CSRT":
		    	tracker = cv2.TrackerCSRT_create()

	
     
    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box


   
    # Initialize tracker with first frame and bounding box
    #ok = tracker.init(frame, bbox)
	# Why do we have this twice?
	#ok, frame = video.read()
 	while True:
		ok, frame = video.read()
	    	while inFrame and personDetected and not SLEEP:
		#last_coord = 300, 300
			tracker.clear()
			tracker = cv2.TrackerKCF_create()
			bbox, name = find_face()
			#print name + " was detected"
			if name != followName:
				#lost_frames = lost_frames + 1
				#if not lost_frames<3:
				if trackFail:
					inFrame = False
				else:
					personDetected = False
				continue
			trackFail = False
			
		 		
			#ask if you are okay because the face cannot be found
		#print bbox[2]
			ok = tracker.init(frame, bbox)
			n=0
			#lost_frames = 0;
	      

			while n<100 and not SLEEP:	
	
			# Read a new frame
	
				ok, frame = video.read()

				#if not ok:
				#    break
				 
				# Start timertracker = cv2.TrackerKCF_create()
				timer = cv2.getTickCount()
				# Update tracker
				ok, bbox = tracker.update(frame)
				x=bbox[0]+bbox[2]/2
				y=bbox[1]+bbox[3]/2
		
				#print x, y

				# Calculate Frames per second (FPS)
				fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
		 
			# Draw bounding box
				if ok:
			    #last_coord = x, y
			    # Tracking success
			    #found = True
					inFrame = True
					#lost_frames = 0
				    	p1 = (int(bbox[0]), int(bbox[1]))
				    	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				    	cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
				    	x_center = 320
				    	y_center = 160
				    	x_window = 65
				    	y_window = 20

			    #print x, y

			    		if not (x_center - x_window) <= x <= (x_center + x_window):	#Make horizontal displacement prioritized

						if x < (x_center - x_window):	# face is left
							arduino.write(struct.pack('>B', 2))

					   	elif x > (x_center + x_window):	# face is right				
							arduino.write(struct.pack('>B', 3))

						else:				# face is straight
							arduino.write(struct.pack('>B', 0))	

			    		else:

				    		if y < (y_center - y_window):	# face is too close
							arduino.write(struct.pack('>B', 4))

				    		elif y > (y_center + y_window):	# face is too far				
							arduino.write(struct.pack('>B', 1))

				    		else:				# face is centered
							arduino.write(struct.pack('>B',0))			
			


				else :
			    # Tracking failure
					#lost_frames = lost_frames + 1
			    		cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
			    		arduino.write(struct.pack('>B', 0))
					n=100
					trackFail = True
					


		 
			# Display result
				cv2.imshow("Tracking", frame)
		 
			# Exit if ESC pressed
				k = cv2.waitKey(1) & 0xff
				if k == 27 : break
				n=n+1
######################################

@ask.launch
def launched():
	global followName
	global inFrame
	global personDetected
	inFrame =True
	personDetected = True
	followName = ''
	return question('Super Robot is listening.')

@ask.intent("identifyIntent")
def response(names):
	global followName
	global inFrame
	global personDetected
	global trackFail
	trackFail = False
	inFrame =True
	personDetected = True
	if names  == 'Rain':
		msg = "Searching for Renooka"
		followName = 'Renooka'
	elif names == 'Eric':
		msg = "Searching for Erick"
		followName = 'Erick'
	elif names == 'Arnold':
		msg = "Searching for Arvand"
		followName = 'Arvand'
	return question(msg)

@ask.intent("OkayIntent")
def okay(response):
	if response == 'yes':
		return question("Okay. Now what, " + followName + "?")
	else:
		message = client.messages.create(body=(followName + " has fallen and needs assistance."), from_="+13016835506", to="+17326107074")
		message.sid
		return statement("I'm contacting emergency services. It'll be okay, " + followName)
	

@ask.intent("MonitorIntent")
def monitor():
	if followName not in TeamNames:
		return question("I have not been told to identify someone.")	
	
	
	
	global inFrame
	global personDetected
	global threadOn
	global SLEEP
	global trackFail
	trackFail = False
	SLEEP = False
	#inFrame = True
	#personDetected = True
	
	if not threadOn:
		try:
			thread.start_new_thread(tracking, ())
			threadOn = True
		except:
			print "Error: unable to start thread"
#while True:
	time.sleep(4)
	if not inFrame and personDetected:
		arduino.write(struct.pack('>B',0))
		inFrame = True
		personDetected = True
		return question("I lost you, " + followName + ". Are you okay?")
	elif not personDetected and inFrame:
		arduino.write(struct.pack('>B',0))
		inFrame = True
		personDetected = True
		return question("I could not find " + followName + ". Now what?")
	return question("Should I keep going?")
		
@ask.intent("moveIntent")
def move(direction):
	global SLEEP
	SLEEP = True
	action = ''
	if direction == 'forwards':
		action = 'I moved '
		arduino.write(struct.pack('>B',5))
		time.sleep(0.25)
		arduino.write(struct.pack('>B',0))
	elif direction == 'backwards':
		action = 'I moved '
		arduino.write(struct.pack('>B',6))
		time.sleep(0.25)
		arduino.write(struct.pack('>B',0))
	elif direction == 'left':
		action = 'I turned '
		arduino.write(struct.pack('>B',7))
		time.sleep(0.25)
		arduino.write(struct.pack('>B',0))
	elif direction == 'right':
		action = 'I turned '
		arduino.write(struct.pack('>B',8))
		time.sleep(0.25)
		arduino.write(struct.pack('>B',0))
	elif direction == 'halt':
		action = 'halting'
		direction = ''
		arduino.write(struct.pack('>B',0))
	return question(action + direction)

@ask.intent("AMAZON.FallbackIntent")
def fallback():
	return question("I didn't understand your request.")

@ask.intent("AMAZON.StopIntent")
def stop():
	global followName
	arduino.write(struct.pack('>B',0))
	name = followName
	followName = ''
	global SLEEP
	SLEEP=True
	if name == '':
		return question("I was not tracking anyone. What should I do now?")
	else:
		return question("I have stopped tracking " + name + ". Now what?")

@ask.intent("sleepIntent")
def sleep():
	arduino.write(struct.pack('>B',0))
	global SLEEP
	SLEEP=True
	return statement("Super Robot is going to sleep.")

@ask.intent("danceIntent")
def dance():
	arduino.write(struct.pack('>B',0))
	global SLEEP
	SLEEP=True
	url = "https://youtu.be/QfPg_GzC-HA"
	video = pafy.new(url)
	best = video.getbest()
	playurl = best.url
	arduino.write(struct.pack('>B',9))
	time.sleep(0.25)
	arduino.write(struct.pack('>B',0))
	return audio().play(playurl, offset=40000)


app.run()









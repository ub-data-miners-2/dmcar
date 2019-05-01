# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from collections import deque
from imutils.video import VideoStream
from logic import Logic
import numpy as np
import argparse
import cv2
import imutils
import time
from picar import back_wheels, front_wheels
import picar
from Line import Line
from lane_detection import color_frame_pipeline
from lane_detection import PID
import time
import math
import os

class DmCar:
	MAX_ANGLE = 20			# Maximum angle to turn right at one time
	MIN_ANGLE = -MAX_ANGLE		# Maximum angle to turn left at one time

	def __init__(self, config="/home/pi/dmcar/picar/config"):
		picar.setup()
		fw = front_wheels.Front_Wheels(debug=False, db=config)
		fw.turn(90) # set straight
		bw = back_wheels.Back_Wheels(debug=False, db=config)

		bw.ready()
		fw.ready()
		self.front_wheels = fw
		self.back_wheels = bw
		self.is_moving = False

	def stop(self):
		self.back_wheels.stop()
		self.is_moving = False

	def forward(self):
		self.back_wheels.forward()
		self.is_moving = True

	def backward(self):
		self.back_wheels.backward()
		self.is_moving = True

	def turn(self, angle):
		self.front_wheels.turn(angle)

	def set_speed(self, amount):
		self.back_wheels.speed = amount

class Camera:
	def __init__(self, src = 0):
		self.stream = VideoStream(src=src).start()

	def off(self):
		self.stream.stop()

	def get_frame(self):
		frame = self.stream.read()
		frame = imutils.resize(frame, width=320)
		(h, w) = frame.shape[:2]
		r = 320 / float(w)
		dim = (320, int(h * r))
		frame = cv2.resize(frame, dim, cv2.INTER_AREA)
		# resize to 320 x 180 for wide frame
		frame = frame[0:180, 0:320]
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return frame

	def get_section(self, frame, x1=230, y1=100, x2=286, y2=156):
		return [frame[y1:y2, x1:x2], (x1, y1), (x2, y2)]


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip, e.g., -v out_video.mp4")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# detect lane based on the last # of frames
frame_buffer = deque(maxlen=args["buffer"])
camera = Camera()
# allow the camera or video file to warm up
time.sleep(2.0)
car = DmCar()
brain = Logic()

# keep looping
while True:
	frame = camera.get_frame()
	frame_buffer.append(frame)
	blend_frame, lane_lines = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)

	# prepare the image to be classified by our deep learning network
	image1, p1, p2 = camera.get_section(frame)
	image = cv2.resize(image1 , (32, 32))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction
	action, proba = brain.action_to_take(image)
	if action == "stop":
		car.stop()
	elif action == "slow":
		car.set_speed(20)
	else:
		car.set_speed(30)
		car.forward()

	label = action
	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	blend_frame = cv2.rectangle(blend_frame, p1, p2, (0,0,255), 2)
	blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
	blend_frame = cv2.putText(blend_frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow('blend', blend_frame)
	cv2.imshow('input', image1)
	key = cv2.waitKey(1) & 0xFF

camera.off()

# close all windows
cv2.destroyAllWindows()

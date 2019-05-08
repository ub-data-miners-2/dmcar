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
	MAX_ANGLE = 110			# Maximum angle to turn right at one time
	MIN_ANGLE = 70                  # Maximum angle to turn left at one time
	DEFAULT_SPEED = 30
	INITIAL_ANGLE = 90

	def __init__(self, config="/home/pi/dmcar/picar/config"):
		picar.setup()
		fw = front_wheels.Front_Wheels(debug=False, db=config)
		fw.turn(self.INITIAL_ANGLE) # set straight
		bw = back_wheels.Back_Wheels(debug=False, db=config)

		bw.ready()
		fw.ready()
		self.angle = self.INITIAL_ANGLE
		self.front_wheels = fw
		self.back_wheels = bw
		self.started = False
		self.is_moving = False
		self.position_error = []
		self.set_speed(0)

	def start(self):
		self.started = True

	def stop(self):
		self.back_wheels.stop()
		self.is_moving = False

	def forward(self):
		self.set_speed(self.DEFAULT_SPEED)
		self.back_wheels.forward()
		self.is_moving = True

	def backward(self):
		self.back_wheels.backward()
		self.is_moving = True

	def turn(self, angle):
		self.angle = angle
		self.front_wheels.turn(angle)

	def set_speed(self, amount):
		self.back_wheels.speed = amount

	def adjust_position(self, w, h, lane_lines):
		y2L = h - 1
		x2L = int((y2L - lane_lines[0].bias) / lane_lines[0].slope)
		y2R = h - 1
		x2R = int((y2R - lane_lines[1].bias) / lane_lines[1].slope)
		mid_position_lane = ( x2R + x2L ) / 2
		self.lane_middle = mid_position_lane

		if not self.is_moving:
			return

		# negative -> + ANGLE, positive -> - ANGLE
		car_position_err = w/2 - mid_position_lane
		car_position_time = time.time()
		self.position_error.append([car_position_err, car_position_time])

		# Control Car
		# Adjust P(KP), I(KI), D(KD) values as well as portion
		# angle = PID(posError, KP=0.8, KI=0.05, KD=0.1) * 0.2
		delta_angle = PID(self.position_error, KP=0.05, KI=0.1, KD=1.5) * 0.2
		print(delta_angle)

		new_angle = self.angle - delta_angle

		# Right turn max 110, Left turn max 70
		if new_angle >= self.MAX_ANGLE:
			new_angle = self.MAX_ANGLE
		elif new_angle <= self.MIN_ANGLE:
			new_angle = self.MIN_ANGLE

		self.turn(new_angle)

class Camera:
	def __init__(self, src = 0):
		self.stream = VideoStream(src=src).start()

	def off(self):
		self.stream.stop()

	def get_frame(self):
		frame = self.stream.read()
		frame = imutils.resize(frame, width=320)
		#(h, w) = frame.shape[:2]
		#r = 320 / float(w)
		#dim = (320, int(h * r))
		#frame = cv2.resize(frame, dim, cv2.INTER_AREA)
		# resize to 320 x 180 for wide frame
		#frame = frame[0:180, 0:320]
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return frame

	def get_section(self, frame, x1=230, y1=100, x2=320, y2=180):
		return [frame[y1:y2, x1:x2], (x1, y1), (x2, y2)]


def handleKeyPress(key, car):
	keycmd = chr(key)

	# if the 'q' key is pressed, end program
	# if the 'w' key is pressed, moving forward
	# if the 'x' key is pressed, moving backword
	# if the 'a' key is pressed, turn left
	# if the 'd' key is pressed, turn right
	# if the 's' key is pressed, straight
	# if the 'z' key is pressed, stop a car
	if keycmd == 'q':
		camera.off()
		car.stop()
		cv2.destroyAllWindows()
		exit()
	elif keycmd == 'w':
		car.start()
		car.set_speed(30)
		car.forward()
	elif keycmd == 'x':
		car.backward()
	elif keycmd == 'a':
		angle -= 5
		if angle <= 45:
			angle = 45
		car.turn(angle)
	elif keycmd == 'd':
		angle += 5
		if angle >= 135:
			angle = 135
		car.turn(angle)
	elif keycmd == 's':
		car.turn(90)
	elif keycmd == 'z':
		car.stop()

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
	# keep straight
	h, w = frame.shape[:2]
	car.adjust_position(w, h, lane_lines)

	# prepare the image to be classified by our deep learning network
	c1 = car.lane_middle + 70
	if c1 > 280:
		c1 = 280
	c2 = c1 + 90
	if c2 > 320:
		c2 = 320
	image1, p1, p2 = camera.get_section(frame,x1=int(c1),x2=int(c2))
	image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image1 , (32, 32))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction
	action, sign, proba = brain.action_to_take(image)
	if action == "stop":
		car.stop()
	elif action == "slow":
		car.set_speed(20)
	else:
		if car.started():
			car.forward()

	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(sign, proba * 100)
	blend_frame = cv2.rectangle(blend_frame, p1, p2, (0,0,255), 2)
	blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
	blend_frame = cv2.putText(blend_frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow('blend', blend_frame)
	cv2.imshow('input', image1)
	key = cv2.waitKey(1) & 0xFF
	handleKeyPress(key, car)

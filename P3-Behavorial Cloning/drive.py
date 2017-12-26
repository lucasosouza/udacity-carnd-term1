import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from preprocessing import process_image, center_scale, process_image_nvidia

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # img = image_array[None, :, :, :]

    # transform image
    img = process_image(image_array).reshape(-1,16,32,3)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = calc_angle(img)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = .25

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


def calc_angle(image):
    """ Get average from the each cameras classifier """
    return float(model.predict(image, batch_size=1))

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(), # converts float to string
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':

    # load model
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
        # compile
        model.compile("adam", "mse")
        # load weights
        model.load_weights('model.h5')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import cv2

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10  # Increased for better control

def img_preprocessing(img):
    # Remove mpimg.imread - img is already a NumPy array
    img = img[60:135,:,:]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Color conversion
    img = cv2.GaussianBlur(img, (3,3), 0)  # Blur
    img = cv2.resize(img, (200,66))  # Resize
    img = img/255  # Normalize
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        print(f"Received telemetry: Speed={speed}")
        
        # Handle image data
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = np.asarray(image)
            image = img_preprocessing(image)  # Directly pass the array
            image = np.array([image])  # Add batch dimension
            
            # Predict steering
            steering_angle = float(model.predict(image, batch_size=100))
            throttle = 1.0 - speed/speed_limit  # Dynamic throttle
            
            # Clamp steering values
            steering_angle = np.clip(steering_angle, -1.0, 1.0)
            
            print(f"Predicted steering: {steering_angle:.4f}, Throttle: {throttle:.4f}")
            send_control_message(steering_angle, throttle)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            send_control_message(0.0, 0.0)
    else:
        print("Empty telemetry received")
        send_control_message(0.0, 0.0)

@sio.on('connect')
def connect(sid, environ):
    print(f'Client connected: {sid}')
    send_control_message(0.0, 0.0)

def send_control_message(steering_angle, throttle):
    message = {
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }
    sio.emit('steer', data=message)
    print(f"Sent: angle={steering_angle:.4f}, throttle={throttle}")

if __name__ == '__main__':
    try:
        model = load_model('compatible_model.h5', compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        exit(1)
    
    # Warm-up the model
    dummy_img = np.zeros((66, 200, 3))
    model.predict(np.array([dummy_img]))
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app, log_output=False)
    print("Server running on port 4567")
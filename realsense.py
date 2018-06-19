from __future__ import print_function

import numpy as np
import cv2
## setup logging
import logging
logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs

def convert_z16_to_bgr(frame):
    '''Performs depth histogram normalization

    This raw Python implementation is slow. See here for a fast implementation using Cython:
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/cython_methods/methods.pyx
    '''
    hist = np.histogram(frame, bins=0x10000)[0]
    hist = np.cumsum(hist)
    hist -= hist[0]
    rgb_frame = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)

    zeros = frame == 0
    non_zeros = frame != 0

    f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
    rgb_frame[non_zeros, 0] = 255 - f
    rgb_frame[non_zeros, 1] = 0
    rgb_frame[non_zeros, 2] = f
    rgb_frame[zeros, 0] = 20
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 0
    return rgb_frame

## start the service - also available as context manager
serv = pyrs.Service()

## create a device from device id and streams of interest
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60), pyrs.stream.DepthStream(fps= 60)])

## retrieve 60 frames of data
while (1):
    cam.wait_for_frames()
    #print(cam.color)
    d = convert_z16_to_bgr(cam.depth)
    cv2.imshow('Depth', d)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


## stop camera and service
cam.stop()
serv.stop()
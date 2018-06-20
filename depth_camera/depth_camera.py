from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import cv2
# setup logging
import logging
logging.basicConfig(level = logging.INFO)

# import the package
import pyrealsense as pyrs

class DepthCamera:
    def __init__(self, fps):
        # start the service - also available as context manager
        self.serv = pyrs.Service()
        self.fps = fps
        # create a device from device id and streams of interest
        self.cam = self.serv.Device(device_id = 0, streams = [pyrs.stream.DepthStream(fps = fps)])


    def __del__(self):
        # stop camera and service
        self.cam.stop()
        self.serv.stop()


    def _convert_z16_to_bgr(self, frame):
        '''
        Performs depth histogram normalization

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
        rgb_frame[zeros, 0] = 0
        rgb_frame[zeros, 1] = 255
        rgb_frame[zeros, 2] = 0

        return rgb_frame


    def _convert_z16_to_z8(self, frame):
        hist = np.histogram(frame, bins=0x10000)[0]
        hist = np.cumsum(hist)
        hist -= hist[0]
        z8_frame = np.empty(frame.shape[:2], dtype=np.uint8)

        zeros = frame == 0
        non_zeros = frame != 0

        f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
        z8_frame[non_zeros] = 255 - f
        z8_frame[zeros] = 0

        return z8_frame


    def get_depth_raw(self):
        self.cam.wait_for_frames()
        d = self.cam.depth * self.cam.depth_scale
        # d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        return d


    def get_depth_normalized_rgb(self):
        self.cam.wait_for_frames()
        return self._convert_z16_to_bgr(self.cam.depth)


    def get_depth_normalized_z8(self):
        self.cam.wait_for_frames()
        return self._convert_z16_to_z8(self.cam.depth)


    def get_central_range(self, half_width):
        self.cam.wait_for_frames()
        d = np.array(self.cam.depth * self.cam.depth_scale)
        x1 = d.shape[0]/2 - half_width
        x2 = d.shape[0]/2 + half_width + 1
        return np.mean(d[x1:x2:1][:], axis=0)
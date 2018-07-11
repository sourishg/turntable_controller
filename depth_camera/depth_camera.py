from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import cv2

# import the package
import pyrealsense2 as rs

class DepthCamera:
    def __init__(self, width, height, fps):
        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            self.width = width
            self.height = height
            self.fps = fps
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            pass


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
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_frame, depth_image


    def get_depth_z8(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_z8 = cv2.convertScaleAbs(depth_image, alpha=0.03)
        return depth_frame, depth_image_z8


    def get_central_range(self, half_width):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        band = []
        if depth_frame:
            for y in range(self.height / 2 - half_width, self.height / 2 + half_width, 1):
                row = []
                for x in range(self.width):
                    d = depth_frame.get_distance(x, y)
                    row.append(d)
                band.append(row)
            band = np.array(band).astype('float32')
        else:
            band = np.zeros(self.width)
        return np.mean(band, axis=0)
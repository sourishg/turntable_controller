from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import cv2

from depth_camera import DepthCamera

cam_width = 640
cam_height = 480

if __name__ == '__main__':
    cam = DepthCamera(60)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(ymin=0, ymax=10)
    scan_plt, = ax.plot([i for i in range(cam_width)], [0 for i in range(cam_width)], 'b-')

    while (1):
        rgb = cam.get_depth_raw()
        scan = cam.get_central_range(5)
        cv2.imshow('Depth', rgb)
        scan_plt.set_ydata(scan)
        fig.canvas.draw()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
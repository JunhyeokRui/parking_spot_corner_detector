"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
#!/home/dyros/anaconda3/envs/yhpark/bin/python
# from torch._C import R
from torch.cuda import set_stream
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Int32, String

import math
import time
import argparse
import sys
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_exact_point
from utils.torch_utils import select_device, load_classifier, time_synchronized
from torchvision.models import resnet18
import torchvision.transforms as T
import numpy as np

bridge = CvBridge()

class after_corner_detection():
    def __init__(self, ):
        
        self.corner_points = None
        self.pub_exact_box_position = rospy.Publisher('/exact_box_position', String, queue_size=10)
 
    def AVM_callback(self, msg):
        # print("Received an image!")
        # try:
            # Convert your ROS Image message to OpenCV2
        # print(msg)
        # t0 = time.time()
        self.img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # t1 = time.time()

        # print('time per detection : {:.5f}'.format(t1-t0))

        blank_image = np.zeros(self.img.shape, np.uint8)
        # print(self.img.shape)
        cv2.circle(blank_image,(200,200), 5, (0,255,255), -1)

        for i in range(self.corner_points.shape[0]):
            cv2.circle(self.img, (self.corner_points[i][0], self.corner_points[i][1]), 3, (0,255,255), 1)
            cv2.circle(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), 3, (0,255,255), 1)

            if self.corner_points.shape[0]==2:
                for j in range(self.corner_points.shape[0]):
                    if i!=j:
                        length = (self.corner_points[i][0]-self.corner_points[j][0])**2 + (self.corner_points[i][1]-self.corner_points[j][1])**2

                        if  length <2000:
                            cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)

                        theta = math.atan2(self.corner_points[i][1]-self.corner_points[j][1],self.corner_points[i][0]-self.corner_points[j][0])
                        theta = math.degrees(theta)

                        cv2.HoughLines(self.img, 0.5, int(theta), 100)


            # if self.corner_points.shape[0]==3:
            #     for j in range(self.corner_points.shape[0]):
            #         if i!=j:
            #             length = (self.corner_points[i][0]-self.corner_points[j][0])**2 + (self.corner_points[i][1]-self.corner_points[j][1])**2

            #             if length <2000:
            #                 cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)
            #             elif 5000 < length and length < 6000:
            #                 cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 0, 255), thickness=2, lineType=8)

            # if self.corner_points.shape[0]==4:
            #     for j in range(self.corner_points.shape[0]):
            #         if i!=j:
            #             length = (self.corner_points[i][0]-self.corner_points[j][0])**2 + (self.corner_points[i][1]-self.corner_points[j][1])**2

            #             if length <2000:
            #                 cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)
            #                 short_slant = (self.corner_points[i][0]-self.corner_points[j][0]) / (self.corner_points[i][1]-self.corner_points[j][1])

            # if self.corner_points.shape[0]==4:
            #     for j in range(self.corner_points.shape[0]):
            #         if i!=j:
            #             length = (self.corner_points[i][0]-self.corner_points[j][0])**2 + (self.corner_points[i][1]-self.corner_points[j][1])**2
            #             this_slant = (self.corner_points[i][0]-self.corner_points[j][0]) / (self.corner_points[i][1]-self.corner_points[j][1])
            #             if length <2000:
            #                 cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)

            #             elif 5000 < length and length < 6000:
            #                 if abs(this_slant * short_slant + 1)<0.1:
            #                     cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 0, 255), thickness=2, lineType=8)

        #         for j in range(self.corner_points.shape[0]):
        #             if i!=j:
        #                 slant = (self.corner_points[j][1]-self.corner_points[i][1]) / (self.corner_points[j][0]-self.corner_points[i][0])
        #                 slants.append(slant)

        #     for i in range(self.corner_points.shape[0]):
        #         for j in range(self.corner_points.shape[0]):
        #             if i!=j:
        #                 this_slant = (self.corner_points[j][1]-self.corner_points[i][1]) / (self.corner_points[j][0]-self.corner_points[i][0])
        #                 for k in slants:
        #                     if k * this_slant +1 < 0.01:
        #                         cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)
        #                         pass
        # else:
        #     for i in range(self.corner_points.shape[0]):
        #         cv2.circle(self.img, (self.corner_points[i][0], self.corner_points[i][1]), 3, (0,255,255), 1)
        #         cv2.circle(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), 3, (0,255,255), 1)
        #         for j in range(self.corner_points.shape[0]):
        #             if i!=j:
        #                 cv2.line(blank_image, (self.corner_points[i][0], self.corner_points[i][1]), (self.corner_points[j][0], self.corner_points[j][1]), (0, 255, 0), thickness=2, lineType=8)
 
        cv2.imshow("Center AVM", self.img)
        cv2.imshow("Only the Corner Points", blank_image)

        # gray = cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY)
        # ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # dst = blank_image.copy()
        # for i in contours:
        #     hull = cv2.convexHull(i, clockwise=True)
        #     cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)



        #  = cv2.convexHull(contour)
        # cv2.drawContours(img1, [hull], 0, (0,0,255), 3)
        # cv2.imshow("convexHull",img1)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # hull = cv2.convexHull(i, clockwise=True)
        # cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)


        # gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

        # kernel_size = 5
        # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


        # cv2.imshow("Edges", edges)
        
        
        cv2.waitKey(1)



    def string_callback(self, msg):
        # print("Received an image!")
        # try:
            # Convert your ROS Image message to OpenCV2
        # print(msg)
        self.corner_points = np.array(self.string_to_numpy(msg.data)).reshape(-1,2)
        # print(self.corner_points)



    def string_to_numpy(self,input_string):
        
        return list(map(int, input_string.split(',')))


    def listener(self):
        rospy.init_node('after_corner_detection')
        rospy.Subscriber("/exact_corner_points_on_AVM", String, self.string_callback)
        rospy.Subscriber("/AVM_center_image", Image, self.AVM_callback)
        rospy.spin()


def parse_opt():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    return opt


def main(opt):

    After_corner_detection = after_corner_detection(**vars(opt))
    After_corner_detection.listener()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

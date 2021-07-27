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
    def __init__(self):
        
        self.corner_points = None
        self.pub_exact_box_position = rospy.Publisher('/exact_box_position', String, queue_size=10)
        self.colors = [(0,255,255),(0,0,255), (255,255,255)]
        self.class_dict = {
            'outside':0,
            'inside':1,
            'middle':2,
        }

    def length_between_points(self,point1,point2):
        return np.linalg.norm(point1 - point2)**2

    def angle_between_points(self,point1,point2):
        return np.arctan((point1 - point2)[1]/(point1 - point2)[0])

    def center_point(self,point1,point2):
        return (point1+point2)/2

    def draw_line(self,image,point1,point2,classes):
        if type(image)==list:
            for imgs in image:
                cv2.line(imgs, tuple(point1), tuple(point2), self.colors[self.class_dict[classes]],2,1)
        else:
            cv2.line(image, tuple(point1), tuple(point2), self.colors[self.class_dict[classes]],2,1)


    def orth_angle_calc(self,angle, direction):
        orth_angle = angle + np.pi/2
        if direction == 'up':
            if np.sin(orth_angle)<0:
                orth_angle = orth_angle - np.pi
        elif direction == 'down':
            if np.sin(orth_angle)>0:
                orth_angle = orth_angle - np.pi
        elif direction == 'left':
            if np.cos(orth_angle)<0:
                orth_angle = orth_angle - np.pi
        elif direction == 'right':
            if np.cos(orth_angle)>0:
                orth_angle = orth_angle - np.pi

        return orth_angle

    def angle_converter(self,angle, direction):
        if direction == 'up':
            if np.sin(angle)<0:
                angle = angle - np.pi
        elif direction == 'down':
            if np.sin(angle)>0:
                angle = angle - np.pi
        elif direction == 'left':
            if np.cos(angle)<0:
                angle = angle - np.pi
        elif direction == 'right':
            if np.cos(angle)>0:
                angle = angle - np.pi

        return angle


    def orth_opposite_point_calc(self, point1, point2, orth_angle, length = 95):
        orth_unit_vec = np.array([np.cos(orth_angle), np.sin(orth_angle)])
        orth_point1 = tuple(list(map(int,point1 - length*orth_unit_vec)))
        orth_point2 = tuple(list(map(int,point2 - length*orth_unit_vec)))

        return orth_point1, orth_point2

    def unit_vector_mul(self, angle1, angle2):
        vec1 = np.array([np.cos(angle1), np.sin(angle1)])
        vec2 = np.array([np.cos(angle2), np.sin(angle2)])

        return np.dot(vec1,vec2)


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

        inside_corner_points = []
        outside_corner_points = []

        inside_corner_points_length = []
        outside_corner_points_length = []
        in_cnt = 0
        out_cnt =0


        for i in range(self.corner_points.shape[0]):
            cv2.circle(self.img, (self.corner_points[i][1], self.corner_points[i][2]), 3, self.colors[self.corner_points[i][0]], -1)
            cv2.circle(blank_image, (self.corner_points[i][1], self.corner_points[i][2]), 3, self.colors[self.corner_points[i][0]], -1)

            if self.corner_points[i][0]==0:
                if self.corner_points[i][2]>200:
                    outside_corner_points.append(np.array([self.corner_points[i][1], self.corner_points[i][2]]))
                    outside_corner_points_length.append((200-self.corner_points[i][1])**2 + (200-self.corner_points[i][2])**2)
                    out_cnt +=1 
            else:
                if self.corner_points[i][2]>200:
                    inside_corner_points.append(np.array((self.corner_points[i][1], self.corner_points[i][2])))
                    inside_corner_points_length.append((200-self.corner_points[i][1])**2 + (200-self.corner_points[i][2])**2)
                    in_cnt +=1
        
        outside_pairs = dict()
        n_outside_corner_pairs =0
        outside_pair_closest_to_center = 10e8

        for i in range(len(outside_corner_points)):
            for j in range(len(outside_corner_points)):
                if not i<j:
                    if self.length_between_points(outside_corner_points[i],outside_corner_points[j])<3000 and self.length_between_points(outside_corner_points[i],outside_corner_points[j])>1000:
                        n_outside_corner_pairs+=1 
                        outside_pairs[n_outside_corner_pairs] = dict()
                        outside_pairs[n_outside_corner_pairs]['corner1'] = outside_corner_points[i]
                        outside_pairs[n_outside_corner_pairs]['corner2'] = outside_corner_points[j]
                        outside_pairs[n_outside_corner_pairs]['length'] = self.length_between_points(outside_corner_points[i],outside_corner_points[j])
                        outside_pairs[n_outside_corner_pairs]['angle'] = self.angle_between_points(outside_corner_points[i],outside_corner_points[j])
                        outside_pairs[n_outside_corner_pairs]['center_point'] = self.center_point(outside_corner_points[i],outside_corner_points[j])
                        outside_pairs[n_outside_corner_pairs]['distance_from_center'] = self.length_between_points(outside_pairs[n_outside_corner_pairs]['center_point'],np.array([200,200]))
                        if outside_pairs[n_outside_corner_pairs]['distance_from_center'] < outside_pair_closest_to_center:
                            outside_pair_closest_to_center = outside_pairs[n_outside_corner_pairs]['distance_from_center']
                            outside_pairs['closest_pair_index'] = n_outside_corner_pairs 

                    # self.draw_line(blank_image, outside_corner_points[i], outside_corner_points[j], 'outside')

        inside_pairs = dict()
        n_inside_corner_pairs =0
        inside_pair_closest_to_center = 10e8
        for i in range(len(inside_corner_points)):
            for j in range(len(inside_corner_points)):
                if i<j:
                    if self.length_between_points(inside_corner_points[i],inside_corner_points[j])<3000 and self.length_between_points(inside_corner_points[i],inside_corner_points[j])>1000:
                        n_inside_corner_pairs+=1 
                        inside_pairs[n_inside_corner_pairs] = dict()
                        inside_pairs[n_inside_corner_pairs]['corner1'] = inside_corner_points[i]
                        inside_pairs[n_inside_corner_pairs]['corner2'] = inside_corner_points[j]
                        inside_pairs[n_inside_corner_pairs]['length'] = self.length_between_points(inside_corner_points[i],inside_corner_points[j])
                        inside_pairs[n_inside_corner_pairs]['angle'] = self.angle_between_points(inside_corner_points[i],inside_corner_points[j])
                        inside_pairs[n_inside_corner_pairs]['center_point'] = self.center_point(inside_corner_points[i],inside_corner_points[j])
                        inside_pairs[n_inside_corner_pairs]['distance_from_center'] = self.length_between_points(inside_pairs[n_inside_corner_pairs]['center_point'],np.array([200,200]))
                        if inside_pairs[n_inside_corner_pairs]['distance_from_center'] < inside_pair_closest_to_center:
                            inside_pair_closest_to_center = inside_pairs[n_inside_corner_pairs]['distance_from_center']
                            inside_pairs['closest_pair_index'] = n_inside_corner_pairs 
                    # self.draw_line(blank_image, inside_corner_points[i], inside_corner_points[j], 'inside')
                    # print('inside line length : {}'.format(inside_pairs[n_inside_corner_pairs]['length']))


        if n_inside_corner_pairs>0 and n_outside_corner_pairs ==0:
            inside_pair_idx = inside_pairs['closest_pair_index']
            if out_cnt ==1 :
                print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner1']))))
                print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner2']))))
                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx]['corner1'])))<0.2:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx]['corner2'])))<0.2:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None

                if match_idx is not None:
                    self.draw_line([blank_image, self.img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, self.img], outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    temp_angle = self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'up')
                    temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])))
                    self.draw_line([blank_image, self.img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, self.img], temp_point, outside_corner_points[0], 'outside')


                print(match_idx)

            if out_cnt ==0 or match_idx is None:
                orth_angle = self.orth_angle_calc(inside_pairs[inside_pair_idx]['angle'], 'up')
                orth_opposite_1, orth_opposite_2 = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], orth_angle)

                self.draw_line([blank_image, self.img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                self.draw_line([blank_image, self.img], orth_opposite_1, orth_opposite_2, 'outside')
                self.draw_line([blank_image, self.img], orth_opposite_1, inside_pairs[inside_pair_idx]['corner1'], 'middle')
                self.draw_line([blank_image, self.img], orth_opposite_2, inside_pairs[inside_pair_idx]['corner2'], 'middle')


        if n_outside_corner_pairs>0 and n_inside_corner_pairs ==0:
            outside_pair_idx = outside_pairs['closest_pair_index']

            if in_cnt ==1 :
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner1']))))
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner2']))))
                if abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'], self.angle_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx]['corner1'])))<0.1:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'], self.angle_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx]['corner2'])))<0.1:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None

                if match_idx is not None:
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, self.img], inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx], 'middle')
                    temp_angle = self.angle_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'down')
                    temp_point,_ = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx][non_match_idx],outside_pairs[outside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])))
                    self.draw_line([blank_image, self.img], temp_point, outside_pairs[outside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, self.img], temp_point, inside_corner_points[0], 'inside')

                print(match_idx)

            if in_cnt ==0 or match_idx is None:

                if abs(np.cos(self.angle_between_points(outside_pairs[outside_pair_idx]['corner1'],outside_pairs[outside_pair_idx]['corner2'])))<0.1:
                    if outside_pairs[outside_pair_idx]['corner1'][0]>200 and outside_pairs[outside_pair_idx]['corner2'][0]>200:
                        orth_angle = self.orth_angle_calc(outside_pairs[outside_pair_idx]['angle'], 'right')
                    if outside_pairs[outside_pair_idx]['corner1'][0]<200 and outside_pairs[outside_pair_idx]['corner2'][0]<200:
                        orth_angle = self.orth_angle_calc(outside_pairs[outside_pair_idx]['angle'], 'left')
                else:
                    orth_angle = self.orth_angle_calc(outside_pairs[outside_pair_idx]['angle'], 'down')
                orth_opposite_1, orth_opposite_2 = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], orth_angle)

                self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                self.draw_line([blank_image, self.img], orth_opposite_1, orth_opposite_2, 'inside')
                self.draw_line([blank_image, self.img], orth_opposite_1, outside_pairs[outside_pair_idx]['corner1'], 'middle')
                self.draw_line([blank_image, self.img], orth_opposite_2, outside_pairs[outside_pair_idx]['corner2'], 'middle')

        if n_outside_corner_pairs>0 and n_inside_corner_pairs >0:
            outside_pair_idx = outside_pairs['closest_pair_index']
            inside_pair_idx = inside_pairs['closest_pair_index']
            # print(abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'],self.angle_between_points(inside_pairs[inside_pair_idx]['center_point'],outside_pairs[outside_pair_idx]['center_point']))))
            if abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'],self.angle_between_points(inside_pairs[inside_pair_idx]['center_point'],outside_pairs[outside_pair_idx]['center_point'])))<0.1:
                if abs(self.unit_vector_mul(self.angle_between_points(outside_pairs[outside_pair_idx]['corner1'],inside_pairs[inside_pair_idx]['corner1']), self.angle_between_points(outside_pairs[outside_pair_idx]['corner2'],inside_pairs[inside_pair_idx]['corner2'])) -1 )<0.01:
                    pair_type = True
                else:
                    pair_type = False

                if pair_type :
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, self.img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner1'], 'middle')
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner2'], inside_pairs[inside_pair_idx]['corner2'], 'middle')

                else:
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, self.img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner2'], inside_pairs[inside_pair_idx]['corner1'], 'middle')
                    self.draw_line([blank_image, self.img], outside_pairs[outside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'middle')
            else:
                that_one_outside_corner_idx = np.array(outside_corner_points_length).argsort()[0]
                print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner1']))))
                print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner2']))))

                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner1'])))<0.1:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner2'])))<0.1:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None
                print(match_idx)

                if match_idx is not None:
                    self.draw_line([blank_image, self.img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, self.img], outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    temp_angle = self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'up')
                    temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])))
                    self.draw_line([blank_image, self.img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, self.img], temp_point, outside_corner_points[that_one_outside_corner_idx], 'outside')


        # if in_cnt >1 : 
        #     inside_corner_points_length = np.array(inside_corner_points_length)
        #     in_idx1, in_idx2 = inside_corner_points_length.argsort()[0], inside_corner_points_length.argsort()[1]
        #     # for i in range(len(inside_corner_points)):
        #     in_length = (inside_corner_points[in_idx1][0]-inside_corner_points[in_idx2][0])**2 + (inside_corner_points[in_idx1][1]-inside_corner_points[in_idx2][1])**2
        #     print('inside corner point length = {}'.format(in_length))
        #     in_angle = np.arctan((inside_corner_points[in_idx1][1]-inside_corner_points[in_idx2][1])/(inside_corner_points[in_idx1][0]-inside_corner_points[in_idx2][0]))
        #     # in_angle = np.arctan(np.array(inside_corner_points[in_idx1]),np.array(inside_corner_points[in_idx2]))

        #     print('inside_line_angle = {}'.format(in_angle))
        #     orth_in_angle = in_angle + np.pi/2
            
        #     if np.sin(orth_in_angle)<0:
        #         orth_in_angle = orth_in_angle - np.pi
        #     # print(orth_in_angle)
        #     orth_unit_vec = np.array([np.cos(orth_in_angle), np.sin(orth_in_angle)])

        #     orth_out_point_1 = tuple(list(map(int,np.array(inside_corner_points[in_idx1]) - 90*orth_unit_vec)))
        #     orth_out_point_2 = tuple(list(map(int,np.array(inside_corner_points[in_idx2]) - 90*orth_unit_vec)))

        #     print(orth_out_point_1)

        #     if 1000< in_length and in_length < 4000:
        #         cv2.line(blank_image, inside_corner_points[in_idx1], inside_corner_points[in_idx2], self.colors[1],2,1)
        #         cv2.line(self.img, inside_corner_points[in_idx1], inside_corner_points[in_idx2], self.colors[1],2,1)
        #         if out_cnt < 2 :
        #             cv2.line(blank_image, orth_out_point_1, orth_out_point_2, self.colors[0],2,1)
        #             cv2.line(blank_image, orth_out_point_1, inside_corner_points[in_idx1], self.colors[2],2,1)
        #             cv2.line(blank_image, orth_out_point_2, inside_corner_points[in_idx2], self.colors[2],2,1)
        #             cv2.line(self.img, orth_out_point_1, orth_out_point_2, self.colors[0],2,1)
        #             cv2.line(self.img, orth_out_point_1, inside_corner_points[in_idx1], self.colors[2],2,1)
        #             cv2.line(self.img, orth_out_point_2, inside_corner_points[in_idx2], self.colors[2],2,1)

        # if out_cnt >1 :
        #     outside_corner_points_length = np.array(outside_corner_points_length)
        #     out_idx1, out_idx2 = outside_corner_points_length.argsort()[0], outside_corner_points_length.argsort()[1]
        #     # for i in range(len(outside_corner_points)):
        #     out_length = (outside_corner_points[out_idx1][0]-outside_corner_points[out_idx2][0])**2 + (outside_corner_points[out_idx1][1]-outside_corner_points[out_idx2][1])**2 
        #     print('outside corner point length = {}'.format(out_length))

        #     out_angle = np.arctan((outside_corner_points[out_idx1][1]-outside_corner_points[out_idx2][1])/(outside_corner_points[out_idx1][0]-outside_corner_points[out_idx2][0]))
        #     # in_angle = np.arctan(np.array(inside_corner_points[in_idx1]),np.array(inside_corner_points[in_idx2]))

        #     print('outside_line_angle = {}'.format(out_angle))
        #     orth_out_angle = out_angle + np.pi/2
            
        #     if np.sin(orth_out_angle)>0:
        #         orth_out_angle = orth_out_angle - np.pi
        #     # print(orth_in_angle)
        #     orth_unit_vec = np.array([np.cos(orth_out_angle), np.sin(orth_out_angle)])

        #     orth_in_point_1 = tuple(list(map(int,np.array(outside_corner_points[out_idx1]) - 90*orth_unit_vec)))
        #     orth_in_point_2 = tuple(list(map(int,np.array(outside_corner_points[out_idx2]) - 90*orth_unit_vec)))

        #     if 1000< out_length and out_length < 4000:
        #         cv2.line(blank_image, outside_corner_points[out_idx1], outside_corner_points[out_idx2], self.colors[0],2,1)
        #         cv2.line(self.img, outside_corner_points[out_idx1], outside_corner_points[out_idx2], self.colors[0],2,1)

        #         if in_cnt <3 :
        #             cv2.line(blank_image, orth_in_point_1, orth_in_point_2, self.colors[1],2,1)
        #             cv2.line(blank_image, orth_in_point_1, outside_corner_points[out_idx1], self.colors[2],2,1)
        #             cv2.line(blank_image, orth_in_point_2, outside_corner_points[out_idx2], self.colors[2],2,1)
        #             cv2.line(self.img, orth_in_point_1, orth_in_point_2, self.colors[1],2,1)
        #             cv2.line(self.img, orth_in_point_1, outside_corner_points[out_idx1], self.colors[2],2,1)
        #             cv2.line(self.img, orth_in_point_2, outside_corner_points[out_idx2], self.colors[2],2,1)

                # elif in_cnt >1 : 
                #     inside_corner_points_length = np.array(inside_corner_points_length)
                #     in_idx1, in_idx2 = inside_corner_points_length.argsort()[0], inside_corner_points_length.argsort()[1]

                #     cv2.line(blank_image, inside_corner_points[in_idx1], outside_corner_points[out_idx1], self.colors[2],2,1)
                #     cv2.line(blank_image, inside_corner_points[in_idx2], outside_corner_points[out_idx2], self.colors[2],2,1)


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
        self.corner_points = np.array(self.string_to_numpy(msg.data)).reshape(-1,3)
        print(self.corner_points)



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

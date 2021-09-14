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
    def __init__(self, hough, pad, dot_threshold,  dist_threshold, max_line, max_dist):
        
        self.corner_points = None
        self.pub_exact_box_position = rospy.Publisher('/exact_box_position', String, queue_size=10)
        self.point_colors = [(0,255,255),(0,0,255),(255,255,255),(128,0,255)]
        self.line_colors = [(0,255,255),(0,0,255),(255,255,255),(128,0,255)]
        self.class_dict = {
            'outside':0,
            'inside':1,
            'outside_aux':2,
            'inside_aux':3
        }
        self.max_line = max_line
        self.dot_threshold = dot_threshold
        self.dist_threshold = dist_threshold
        self.pad = pad
        self.max_dist = max_dist

        self.inside_corner_points = []
        self.inside_aux_corner_points = []
        self.outside_corner_points = []
        self.outside_aux_corner_points = []

        self.inside_corner_points_length = []
        self.inside_aux_corner_points_length = []
        self.outside_corner_points_length = []
        self.outside_aux_corner_points_length = []

        self.in_cnt = 0
        self.in_aux_cnt =0
        self.out_cnt =0
        self.out_aux_cnt =0

        self.outside_pairs = dict()
        self.n_outside_corner_pairs =0


    def make_binary_img(self, img):
        new_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,0]<105 or img[i,j,1]>125 or img[i,j,2]<150:
                    new_img[i,j,0]=0
                    new_img[i,j,1]=0
                    new_img[i,j,2]=0
                else:
                    new_img[i,j,0]=255
                    new_img[i,j,1]=255
                    new_img[i,j,2]=255

        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        return gray_img, new_img


    def length(self,point1,point2):
        return np.linalg.norm(point1 - point2)**2

    def angle(self,point1,point2):
        return np.arctan((point1 - point2)[1]/(point1 - point2)[0])

    def center(self,point1,point2):
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

    def init_analyze(corner_points):
        for i in range(corner_points.shape[0]):
            cv2.circle(circle_img, (corner_points[i][1], corner_points[i][2]), 3, self.colors[corner_points[i][0]], -1)
            cv2.circle(blank_image, (corner_points[i][1], corner_points[i][2]), 3, self.colors[corner_points[i][0]], -1)

            if corner_points[i][0]==0:
                if corner_points[i][2]>200:
                    self.outside_corner_points.append(np.array([corner_points[i][1], corner_points[i][2]]))
                    self.outside_corner_points_length.append((200-corner_points[i][1])**2 + (200-self.corner_points[i][2])**2)
                    self.out_cnt +=1 
            elif corner_points[i][0]==1:
                if corner_points[i][2]>200:
                    self.inside_corner_points.append(np.array((corner_points[i][1], corner_points[i][2])))
                    self.inside_corner_points_length.append((200-corner_points[i][1])**2 + (200-corner_points[i][2])**2)
                    self.in_cnt +=1
            elif corner_points[i][0]==2:
                if corner_points[i][2]>200:
                    self.inside_aux_corner_points.append(np.array((corner_points[i][1], corner_points[i][2])))
                    self.inside_aux_corner_points_length.append((200-corner_points[i][1])**2 + (200-corner_points[i][2])**2)
                    self.in_aux_cnt +=1
            elif corner_points[i][0]==3:
                if corner_points[i][2]>200:
                    self.outside_aux_corner_points.append(np.array((corner_points[i][1], corner_points[i][2])))
                    self.outside_aux_corner_points_length.append((200-corner_points[i][1])**2 + (200-corner_points[i][2])**2)
                    self.out_aux_cnt +=1

    def find_pairs(which_corner_points):
        nn=0
        which_pairs = dict()
        which_pair_closest_to_center = 10e8

        for i in range(len(which_corner_points)):
            for j in range(len(which_corner_points)):
                if not i<j:
                    if self.length(which_corner_points[i],which_corner_points[j])<3000 and self.length(which_corner_points[i],which_corner_points[j])>1000:
                        nn+=1 
                        which_pairs[nn] = dict()
                        which_pairs[nn]['corner1'] = which_corner_points[i]
                        which_pairs[nn]['corner2'] = which_corner_points[j]
                        which_pairs[nn]['length'] = self.length(which_corner_points[i],which_corner_points[j])
                        which_pairs[nn]['angle'] = self.angle(which_corner_points[i],which_corner_points[j])
                        which_pairs[nn]['center_point'] = self.center(which_corner_points[i],which_corner_points[j])
                        which_pairs[nn]['distance_from_center'] = self.length(which_pairs[nn]['center_point'],np.array([200,200]))
                        if which_pairs[nn]['distance_from_center'] < which_pair_closest_to_center:
                            which_pair_closest_to_center = which_pairs[nn]['distance_from_center']
                            which_pairs['closest_pair_index'] = nn 
        return which_pairs, nn, which_pair_closest_to_center

                    
                    
    def AVM_callback(self, msg):
        self.img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        circle_img = self.img.copy()
        blank_image = np.zeros(self.img.shape, np.uint8)
        cv2.circle(blank_image,(200,200), 5, (0,255,255), -1)

        init_analyze(self.corner_points)

        outside_pairs, n_outside_corner_pairs, outside_pair_closest_to_center = find_pairs(self.outside_corner_points)
        inside_pairs, n_inside_corner_pairs, inside_pair_closest_to_center = find_pairs(self.inside_corner_points)


        print("inside_pair {} / outside_pair {} / inside_corner {} / outside_corner {} / in_aux {} / out_aux {}".format(n_inside_corner_pairs, n_outside_corner_pairs, self.in_cnt, self.out_cnt, self.in_aux_cnt, self.out_aux_cnt))
        
        
        
        
        #################### if inside (back) corner pair is detected without any outside (front) corner pair, 
            
        if n_inside_corner_pairs>0 and n_outside_corner_pairs ==0:   
            
            inside_pair_idx = inside_pairs['closest_pair_index']   # first get the closest inside pair from the center.
            if self.out_cnt ==1 :  # if there is "single" outside corner 
                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.outside_corner_points[0], inside_pairs[inside_pair_idx]['corner1'])))<0.2:   # and if that single outside-corner and inside-[corner1] is orthogonal to the slope of original inside pair,
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.outside_corner_points[0], inside_pairs[inside_pair_idx]['corner2'])))<0.2:  # or if that single outside-corner and inside-[corner1] is orthogonal to the slope of original inside pair,
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'    # store the index of matching corners. 
                else:   # If none of them is orthogonal, there's no matching index. s
                    match_idx = None

                if match_idx is not None:  # If there's a matching index,  draw lines, assuming parallelogram relationship. 
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], self.outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    if in_aux_cnt >0:
                        _min_matmul = 10e6
                        for in_aux_idx in range(in_aux_cnt):
                            _matmul = abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx][non_match_idx])))
                            if _min_matmul > _matmul:
                                _min_matmul = _matmul
                                matching_aux_idx = in_aux_idx
                        if _min_matmul > 0.2:
                            matching_aux_idx = None

                        temp_angle = self.angle(self.inside_aux_corner_points[matching_aux_idx], inside_pairs[inside_pair_idx][non_match_idx])
                        temp_angle = self.angle_converter (temp_angle, 'up')
                        temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length(self.outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])))

                    if in_aux_cnt ==0 or matching_aux_idx is None:
                        temp_angle = self.angle(self.outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])
                        temp_angle = self.angle_converter (temp_angle, 'up')
                        temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length(outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])))

                    self.draw_line([blank_image, circle_img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, self.outside_corner_points[0], 'outside')

            if self.out_cnt ==0 or match_idx is None:   # If there's no outside corner points at all, or if there's no matching index, 
                
                matching_aux_idx_corner1 = None
                matching_aux_idx_corner2 = None

                if self.in_aux_cnt >0:
                    _min_matmul_corner1 = 10e6
                    _min_matmul_corner2 = 10e6
                    for in_aux_idx in range(self.in_aux_cnt):
                        _matmul_corner1 = abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx]['corner1'])))
                        _matmul_corner2 = abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx]['corner2'])))
                        if _min_matmul_corner1 > _matmul_corner1:
                            _min_matmul_corner1 = _matmul_corner1
                            matching_aux_idx_corner1 = in_aux_idx
                            _dist_corner1 = self.length(self.inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx]['corner1'])
                        if _min_matmul_corner2 > _matmul_corner2:
                            _min_matmul_corner2 = _matmul_corner2
                            matching_aux_idx_corner2 = in_aux_idx
                            _dist_corner2 = self.length(self.inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx]['corner2'])

                    if _min_matmul_corner1 > 0.2 or _dist_corner1 < 1500:
                        matching_aux_idx_corner1 = None
                    if _min_matmul_corner2 > 0.2 or _dist_corner2 < 1500:
                        matching_aux_idx_corner2 = None

                ### Assume strictly orthogonal relationship and draw the lines upward with fixed length. 
                if matching_aux_idx_corner1 is None:
                    orth_angle_1 = self.orth_angle_calc_with_hough(inside_pairs[inside_pair_idx]['angle'], 'up', inside_pairs[inside_pair_idx]['corner1'], opt.hough,'corner1')
                else:
                    temp_angle = self.angle(self.inside_aux_corner_points[matching_aux_idx_corner1], inside_pairs[inside_pair_idx]['corner1'])
                    orth_angle_1 = self.angle_converter (temp_angle, 'up')

                if matching_aux_idx_corner2 is None:
                    orth_angle_2 = self.orth_angle_calc_with_hough(inside_pairs[inside_pair_idx]['angle'], 'up', inside_pairs[inside_pair_idx]['corner2'], opt.hough,'corner2')
                else:
                    temp_angle = self.angle(self.inside_aux_corner_points[matching_aux_idx_corner2], inside_pairs[inside_pair_idx]['corner2'])
                    orth_angle_2 = self.angle_converter (temp_angle, 'up')

                orth_opposite_1, _ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], orth_angle_1)
                _, orth_opposite_2 = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], orth_angle_2)

                self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, orth_opposite_2, 'outside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, inside_pairs[inside_pair_idx]['corner1'], 'middle')
                self.draw_line([blank_image, circle_img], orth_opposite_2, inside_pairs[inside_pair_idx]['corner2'], 'middle')




        if n_outside_corner_pairs>0 and n_inside_corner_pairs ==0:

            outside_pair_idx = outside_pairs['closest_pair_index']

            if self.in_cnt ==1 :
                if abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'], self.angle(self.inside_corner_points[0], outside_pairs[outside_pair_idx]['corner1'])))<0.1:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'], self.angle(self.inside_corner_points[0], outside_pairs[outside_pair_idx]['corner2'])))<0.1:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None

                if match_idx is not None:
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, circle_img], self.inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx], 'middle')
                    temp_angle = self.angle(self.inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'down')
                    temp_point,_ = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx][non_match_idx],outside_pairs[outside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length(self.inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])))
                        
                    self.draw_line([blank_image, circle_img], temp_point, outside_pairs[outside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, self.inside_corner_points[0], 'inside')

                # print(match_idx)

            if self.in_cnt ==0 or match_idx is None:

                if abs(np.cos(self.angle(outside_pairs[outside_pair_idx]['corner1'],outside_pairs[outside_pair_idx]['corner2'])))<0.1:
                    if outside_pairs[outside_pair_idx]['corner1'][0]>200 and outside_pairs[outside_pair_idx]['corner2'][0]>200:
                        orth_angle_1 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'right', outside_pairs[outside_pair_idx]['corner1'], opt.hough,'corner1')
                        orth_angle_2 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'right', outside_pairs[outside_pair_idx]['corner2'], opt.hough,'corner2')

                    if outside_pairs[outside_pair_idx]['corner1'][0]<200 and outside_pairs[outside_pair_idx]['corner2'][0]<200:
                        orth_angle_1 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'left', outside_pairs[outside_pair_idx]['corner1'], opt.hough,'corner1')
                        orth_angle_2 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'left', outside_pairs[outside_pair_idx]['corner2'], opt.hough,'corner2')

                else:
                    orth_angle_1 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'down',outside_pairs[outside_pair_idx]['corner1'] , opt.hough,'corner1')
                    orth_angle_2 = self.orth_angle_calc_with_hough(outside_pairs[outside_pair_idx]['angle'], 'down',outside_pairs[outside_pair_idx]['corner2'] , opt.hough,'corner2')

                orth_opposite_1, _ = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], orth_angle_1)
                _, orth_opposite_2 = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], orth_angle_2)

                self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, orth_opposite_2, 'inside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, outside_pairs[outside_pair_idx]['corner1'], 'middle')
                self.draw_line([blank_image, circle_img], orth_opposite_2, outside_pairs[outside_pair_idx]['corner2'], 'middle')

                
                
        if n_outside_corner_pairs>0 and n_inside_corner_pairs >0:

            outside_pair_idx = outside_pairs['closest_pair_index']
            inside_pair_idx = inside_pairs['closest_pair_index']
            if abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'],self.angle(inside_pairs[inside_pair_idx]['center_point'],outside_pairs[outside_pair_idx]['center_point'])))<0.1:
                if abs(self.unit_vector_mul(self.angle(outside_pairs[outside_pair_idx]['corner1'],inside_pairs[inside_pair_idx]['corner1']), self.angle(outside_pairs[outside_pair_idx]['corner2'],inside_pairs[inside_pair_idx]['corner2'])) -1 )<0.01:
                    pair_type = True
                else:
                    pair_type = False

                if pair_type :
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner1'], 'middle')
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner2'], inside_pairs[inside_pair_idx]['corner2'], 'middle')

                else:
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner2'], inside_pairs[inside_pair_idx]['corner1'], 'middle')
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'middle')
            else:
                that_one_outside_corner_idx = np.array(outside_corner_points_length).argsort()[0]

                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner1'])))<0.1:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner2'])))<0.1:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None

                if match_idx is not None:
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    if self.in_aux_cnt >0:
                        _min_matmul = 10e6
                        for in_aux_idx in range(self.in_aux_cnt):
                            _matmul = abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle(self.inside_aux_corner_points[in_aux_idx], inside_pairs[inside_pair_idx][non_match_idx])))
                            if _min_matmul > _matmul:
                                _min_matmul = _matmul
                                matching_aux_idx = in_aux_idx
                        if _min_matmul > 0.2:
                            matching_aux_idx = None

                        temp_angle = self.angle(self.inside_aux_corner_points[matching_aux_idx], inside_pairs[inside_pair_idx][non_match_idx])
                        temp_angle = self.angle_converter (temp_angle, 'up')
                        temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length(self.outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])))

                    if self.in_aux_cnt ==0 or matching_aux_idx is None:
                        temp_angle = self.angle(self.outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])
                        temp_angle = self.angle_converter (temp_angle, 'up')
                        temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length(self.outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])))

                    self.draw_line([blank_image, circle_img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, self.outside_corner_points[that_one_outside_corner_idx], 'outside')


        cv2.imshow("Center AVM", circle_img)
        cv2.imshow("Only the Corner Points", blank_image)
        
        
        cv2.waitKey(1)



    def string_callback(self, msg):
        self.corner_points = np.array(self.string_to_numpy(msg.data)).reshape(-1,3)

    def string_to_numpy(self,input_string):
        return list(map(int, input_string.split(',')))


    def listener(self):
        rospy.init_node('after_corner_detection')
        rospy.Subscriber("/exact_corner_points_on_AVM", String, self.string_callback)
        rospy.Subscriber("/AVM_center_image", Image, self.AVM_callback)
        rospy.spin()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hough', action='store_true')
    parser.add_argument('--pad', type=int, default = 60)
    parser.add_argument('--dot_threshold', type=float, default = 0.05)
    parser.add_argument('--dist_threshold', type=float, default = 64)
    parser.add_argument('--max_line', type=int, default = 30)
    parser.add_argument('--max_dist', type=int, default = 5)

    opt = parser.parse_args()

    return opt


def main(opt):

    After_corner_detection = after_corner_detection(**vars(opt))
    After_corner_detection.listener()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

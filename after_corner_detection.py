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
        self.colors = [(0,255,255),(0,0,255), (255,255,255)]
        self.class_dict = {
            'outside':0,
            'inside':1,
            'middle':2,
        }
        self.max_line = max_line
        self.dot_threshold = dot_threshold
        self.dist_threshold = dist_threshold
        self.pad = pad
        self.max_dist = max_dist

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

    def orth_angle_calc_with_hough(self,angle, direction, corner, hough, name):
        if hough:
            temp = self.img[corner[1]-self.pad: corner[1]+self.pad, corner[0]-self.pad: corner[0]+self.pad]

            # img_canny = cv2.Canny(temp, 50,100)
            gray_binary, color_binary = self.make_binary_img(temp)
            cv2.imshow("binary image around {}".format(name),color_binary)
            img_canny = cv2.Canny(color_binary, 50,100)
            # cv2.imshow("canny image around {}".format(name),img_canny)
            lines = cv2.HoughLinesP(img_canny, 1,np.pi/360,5, np.array([]), self.max_line, self.max_dist)
            if lines is not None:
                # contrast_img.save(os.path.join(new_path,this_class, '{}_high_contrast.jpeg'.format(this_image[:-5])))
                print_img = temp.copy()
                points = []
                angles = []
                distances = []
                dot_products = []
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                        cv2.line(print_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        orig_vec = np.array([np.cos(angle),np.sin(angle)])
                        this_angle = self.angle_between_points(np.array([x1,y1]),np.array([x2,y2]))
                        orth_vec = np.array([np.cos(this_angle), np.sin(this_angle)])
                        prod = np.abs(np.matmul(orth_vec, orig_vec))
                        dot_products.append(prod)
                        angles.append(this_angle)
                        distances.append(min(self.length_between_points(np.array([x1,y1]), np.array([temp.shape[0]//2,temp.shape[1]//2]) ),self.length_between_points(np.array([x2,y2]), np.array([temp.shape[0]//2,temp.shape[1]//2]) )))

                dot_products = np.array(dot_products)
                angles = np.array(angles)
                distances = np.array(distances)
                cv2.imshow("hough lines around {}".format(name),print_img)
                # dot_thresholding = lambda x: 10e6 if x>self.dot_threshold else x
                # distance_thresholding = lambda x: 10e6 if x>self.dist_threshold else x

                # print(dot_products)
                # dot_products = np.array(list(map(dot_thresholding, dot_products)))
                # print(dot_products)
                # distances = np.array(list(map(distance_thresholding,distances)))
                
                # print(dot_products)
                index = np.argmin(1*dot_products + 0.01*distances)
                print(dot_products[index])
                print(distances[index])
                if dot_products[index] < self.dot_threshold and distances[index] <self.dist_threshold:
                # selected_angle = None
                # for jj, hough_angle in enumerate(angles):

                    # if np.abs(np.matmul(orth_vec, orig_vec))<self.dot_threshold and distances[jj] < self.dist_threshold :
                    selected_angle = angles[index]
                else:
                    selected_angle = None

                if selected_angle is not None:
                    # print(min_matmul)
                    orth_angle = selected_angle
                    print("{}: HOUGH SLOPE IS USED!!\n".format(name))
                    # print(lines)
                    # print(final_index)
                    # print((lines[final_index][0]))
                    # cv2.line(print_img, (lines[final_index][0][0],lines[final_index][0][1]), (lines[final_index][0][2], lines[final_index][0][3]), (255, 255, 0), 10)
                else:
                    print("{}: There are no hough lines that are orthogonal enough. Assuming Strictly Orthogonal Relationship!\n".format(name))
                    orth_angle = angle + np.pi/2
            else:
                print("{}: No hough lines are found. Assuming Strictly Orthogonal Relationship!\n".format(name))
                orth_angle = angle + np.pi/2


        else:
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
        circle_img = self.img.copy()
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
            cv2.circle(circle_img, (self.corner_points[i][1], self.corner_points[i][2]), 3, self.colors[self.corner_points[i][0]], -1)
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
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner1']))))
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner2']))))
                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx]['corner1'])))<0.2:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx]['corner2'])))<0.2:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None

                if match_idx is not None:
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    temp_angle = self.angle_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'up')
                    temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(outside_corner_points[0], inside_pairs[inside_pair_idx][match_idx])))
                    self.draw_line([blank_image, circle_img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, outside_corner_points[0], 'outside')


                # print(match_idx)

            if out_cnt ==0 or match_idx is None:
                orth_angle_1 = self.orth_angle_calc_with_hough(inside_pairs[inside_pair_idx]['angle'], 'up', inside_pairs[inside_pair_idx]['corner1'], opt.hough,'corner1')
                orth_angle_2 = self.orth_angle_calc_with_hough(inside_pairs[inside_pair_idx]['angle'], 'up', inside_pairs[inside_pair_idx]['corner2'], opt.hough,'corner2')

                orth_opposite_1, _ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], orth_angle_1)
                _, orth_opposite_2 = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], orth_angle_2)

                self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, orth_opposite_2, 'outside')
                self.draw_line([blank_image, circle_img], orth_opposite_1, inside_pairs[inside_pair_idx]['corner1'], 'middle')
                self.draw_line([blank_image, circle_img], orth_opposite_2, inside_pairs[inside_pair_idx]['corner2'], 'middle')


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
                    self.draw_line([blank_image, circle_img], outside_pairs[outside_pair_idx]['corner1'], outside_pairs[outside_pair_idx]['corner2'], 'outside')
                    self.draw_line([blank_image, circle_img], inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx], 'middle')
                    temp_angle = self.angle_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'down')
                    temp_point,_ = self.orth_opposite_point_calc(outside_pairs[outside_pair_idx][non_match_idx],outside_pairs[outside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(inside_corner_points[0], outside_pairs[outside_pair_idx][match_idx])))
                    self.draw_line([blank_image, circle_img], temp_point, outside_pairs[outside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, inside_corner_points[0], 'inside')

                # print(match_idx)

            if in_cnt ==0 or match_idx is None:

                if abs(np.cos(self.angle_between_points(outside_pairs[outside_pair_idx]['corner1'],outside_pairs[outside_pair_idx]['corner2'])))<0.1:
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
            # print(abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'],self.angle_between_points(inside_pairs[inside_pair_idx]['center_point'],outside_pairs[outside_pair_idx]['center_point']))))
            if abs(self.unit_vector_mul(outside_pairs[outside_pair_idx]['angle'],self.angle_between_points(inside_pairs[inside_pair_idx]['center_point'],outside_pairs[outside_pair_idx]['center_point'])))<0.1:
                if abs(self.unit_vector_mul(self.angle_between_points(outside_pairs[outside_pair_idx]['corner1'],inside_pairs[inside_pair_idx]['corner1']), self.angle_between_points(outside_pairs[outside_pair_idx]['corner2'],inside_pairs[inside_pair_idx]['corner2'])) -1 )<0.01:
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
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner1']))))
                # print(abs(self.unit_vector_mul(inside_pairs[n_inside_corner_pairs]['angle'], self.angle_between_points(outside_corner_points[0], inside_pairs[n_inside_corner_pairs]['corner2']))))

                if abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner1'])))<0.1:
                    match_idx = 'corner1'
                    non_match_idx = 'corner2'
                elif abs(self.unit_vector_mul(inside_pairs[inside_pair_idx]['angle'], self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx]['corner2'])))<0.1:
                    match_idx = 'corner2'
                    non_match_idx = 'corner1'
                else:
                    match_idx = None
                # print(match_idx)

                if match_idx is not None:
                    self.draw_line([blank_image, circle_img], inside_pairs[inside_pair_idx]['corner1'], inside_pairs[inside_pair_idx]['corner2'], 'inside')
                    self.draw_line([blank_image, circle_img], outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx], 'middle')

                    temp_angle = self.angle_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])
                    temp_angle = self.angle_converter (temp_angle, 'up')
                    temp_point,_ = self.orth_opposite_point_calc(inside_pairs[inside_pair_idx][non_match_idx],inside_pairs[inside_pair_idx][non_match_idx],temp_angle,np.sqrt(self.length_between_points(outside_corner_points[that_one_outside_corner_idx], inside_pairs[inside_pair_idx][match_idx])))
                    self.draw_line([blank_image, circle_img], temp_point, inside_pairs[inside_pair_idx][non_match_idx], 'middle')
                    self.draw_line([blank_image, circle_img], temp_point, outside_corner_points[that_one_outside_corner_idx], 'outside')


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


        cv2.imshow("Center AVM", circle_img)
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

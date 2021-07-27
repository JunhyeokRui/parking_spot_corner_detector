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

class yolo_for_ros():
    def __init__(self, yolo_weight,  # model.pt path(s)
            refinement_weight,
            source,  # file/dir/URL/glob, 0 for webcam
            imgsz,  # inference size (pixels)
            conf_thres,  # confidence threshold
            iou_thres,  # NMS IOU threshold
            max_det,  # maximum detections per image
            resize, 
            device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img,  # show results
            save_txt,  # save results to *.txt
            save_conf,  # save confidences in --save-txt labels
            save_crop,  # save cropped prediction boxes
            nosave,  # do not save images/videos
            classes,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms,  # class-agnostic NMS
            augment,  # augmented inference
            visualize,  # visualize features
            update,  # update all models
            project,  # save results to project/name
            name,  # save results to project/name
            exist_ok,  # existing project/name ok, do not increment
            line_thickness,  # bounding box thickness (pixels)
            hide_labels,  # hide labels
            hide_conf,  # hide confidences
            half,
            refinement):

        self.img = None
        self.yolo_weight = yolo_weight
        self.refinement_weight = refinement_weight
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det 
        self.device = device
        self.view_img = view_img 
        self.save_txt = save_txt 
        self.save_conf = save_conf 
        self.save_crop = save_crop 
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update 
        self.project = project 
        self.name = name
        self.exist_ok = exist_ok 
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels 
        self.hide_conf = hide_conf 
        self.half = half 
        self.resize = resize
        self.refinement = refinement

        print('loading yolo model from {}'.format(yolo_weight))
        # print(yolo_weight)
        self.model = attempt_load(yolo_weight, map_location='cpu')  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model = self.model.cuda()

        # Load refinement model
        print('loading refinement model from {}'.format(os.path.join('./refinement_network_weights',refinement_weight)))
        self.refinement_model = resnet18()
        self.refinement_model.fc = nn.Sequential(nn.Linear(512,2),nn.Sigmoid())
        refinement_state = torch.load(os.path.join('./refinement_network_weights',refinement_weight))
        self.refinement_model.load_state_dict(refinement_state)
        self.refinement_model = self.refinement_model.cuda()
        print('MODEL ALL LOADED!!')


        self.pub = rospy.Publisher('/exact_corner_points_on_AVM', String, queue_size=10)
 
    def image_callback(self, msg):
        # print("Received an image!")
        # try:
            # Convert your ROS Image message to OpenCV2
        # print(msg)
        t0 = time.time()
        self.img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # print(x.shape)
        # print('none1')
        self.run(input_img = self.img, yolo_weight = self.yolo_weight,  # model.pt path(s)
            refinement_weight = self.refinement_weight,
            source = self.source,  # file/dir/URL/glob, 0 for webcam
            imgsz = self.imgsz,  # inference size (pixels)
            conf_thres = self.conf_thres,  # confidence threshold
            iou_thres = self.iou_thres,  # NMS IOU threshold
            max_det = self.max_det,  # maximum detections per image
            device = self.device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img = self.view_img,  # show results
            save_txt = self.save_txt,  # save results to *.txt
            save_conf=self.save_conf,  # save confidences in --save-txt labels
            save_crop=self.save_crop,  # save cropped prediction boxes
            nosave=self.nosave,  # do not save images/videos
            classes=self.classes,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=self.agnostic_nms,  # class-agnostic NMS
            augment=self.augment,  # augmented inference
            visualize=self.visualize,  # visualize features
            update=self.update,  # update all models
            project=self.project,  # save results to project/name
            name=self.name,  # save results to project/name
            exist_ok=self.exist_ok,  # existing project/name ok, do not increment
            line_thickness=self.line_thickness,  # bounding box thickness (pixels)
            hide_labels=self.hide_labels,  # hide labels
            hide_conf=self.hide_conf,  # hide confidences
            half=self.half,
            refinement = self.refinement)

    @torch.no_grad()
    def run(self, input_img, yolo_weight='yolov5s.pt',  # model.pt path(s)
            refinement_weight = None,
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            resize = 50, 
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,
            refinement = False):

        # rospy.init_node('image_listener')
        # # imgsaver=image_saver()
        # # Define your image topic
        # image_topic = "/fisheye_raw_rear"
        # # Set up your subscriber and define its callback
        # rospy.Subscriber(image_topic, Image, image_callback)
        # # print(sub)
        # # Spin until ctrl + c
        # # print(imgs)
        # rospy.spin()
        # # print(imgsaver.image)
        

        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        # set_logging()
        # device = select_device(device)
        # half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model

        # Second-stage classifier
        # classify = True
        # if classify:
        #     modelc = load_classifier(name='resnet50', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

        # Dataloader
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        #     bs = len(dataset)  # batch_size
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride)
        #     bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # if device.type != 'cpu':
        self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        # for path, img, im0s, vid_cap in dataset:
        im0 = input_img
        img = torch.from_numpy(input_img).to(device).unsqueeze(0)
        img = torch.transpose(img, 3,1)
        img = torch.transpose(img, 2,3)
        # print(img)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 81.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        img = T.Resize(imgsz)(img)
        # print(img.shape)

        
        # print(img.shape)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img,
                    augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print('this_is_prediction')
        # print(pred)
        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # print(det)
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            # else:
            #     p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = img.copy() if save_crop else img  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print('before_resizing\n')
                # print(det)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print('after_resizing\n')
                # print(det)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # refined_pixel_estimations = []

                
                # Write results
                refine_estimations = []
                class_estimations = []
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    # print('here')
                    # print(xyxy)
                    center_point = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    # refine_estimation = plot_exact_point(self.resize, self.refinement_model,xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness, refinement=self.refinement)
                    refine_estimations.append(np.array([c,center_point[0],center_point[1]]))
                    # class_estimations.append(c)
                    # print(center_point)
                refine_estimations = np.concatenate(refine_estimations,0)
                refine_estimations = ','.join(map(str,refine_estimations))
                self.pub.publish(refine_estimations)

                
                    # self.pub.publish(refine_estimation)
                    # if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                if self.refinement :
                    name = "YOLO with Refinement"
                else:
                    name = "YOLO only"
                cv2.imshow(name,im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
                # else:  # 'video' or 'stream'
                #     if vid_path[i] != save_path:  # new video
                #         vid_path[i] = save_path
                #         if isinstance(vid_writer[i], cv2.VideoWriter):
                #             vid_writer[i].release()  # release previous video writer
                #         if vid_cap:  # video
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         else:  # stream
                #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path += '.mp4'
                #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #     vid_writer[i].write(im0)

        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")

        # if update:
        #     strip_optimizer(yolo_weight)  # update model (to fix SourceChangeWarning)

        # print(f'Done. ({time.time() - t0:.3f}s)')

        t1 = time.time()

        print('time per detection : {:.5f}'.format(t1-t0))

    def listener(self):
        rospy.init_node('yolo_detector')
        # Define your image topic
        # Set up your subscriber and define its callback
        rospy.Subscriber("/AVM_center_image", Image, self.image_callback)
        # rospy.Subscriber("/fisheye_raw_center", Image, self.image_front_callback)
        # rospy.Subscriber("/fisheye_raw_front", Image, self.image_callback)
        # rospy.Publisher()
        rospy.spin()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weight', nargs='+', type=str, default='./runs/train/avm_train_on_ABC/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--refinement_weight', nargs='+', type=str, default='train_ABC_resize_80_crop_45_55_20_resnet18/200_epoch_model.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--resize', type=int, default=50, help='maximum detections per image')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--refinement', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    Yolo_for_ros = yolo_for_ros(**vars(opt))
    Yolo_for_ros.listener()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

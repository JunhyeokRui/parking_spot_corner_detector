import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse


# Picture path

def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(0)
        cv2.circle(img, (x, y), 3, (0, 255, 255), thickness=-1)
        cv2.rectangle(img, (x-30, y-30),(x+30,y+30),  (0, 255, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)
        # return x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(1)
        cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=-1)
        cv2.rectangle(img, (x-30, y-30),(x+30,y+30),  (0, 255, 0), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)
        # return x, y


parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--trial', default=None)
parser.add_argument('--place', default=None)
parser.add_argument('--aug', default=None)

args = parser.parse_args()

path = '/home/dyros/yhpark/new_refinement_data/'

# two_inside_corners = open(os.path.join('position_labels','two_inside_corners.txt'),'w')
# two_outside_corners.seek(0)
# 'trial'

# # if trial is None:
# for trial in os.listdir(path):
#     if args.trial is not None:
#         if trial == args.trial:
#             if os.path.isdir(os.path.join(path,trial)):
#                 for classes in os.listdir(os.path.join(path,trial)):
                    # if 'jpeg' not in classes and os.path.isdir(os.path.join(path,trial,classes)) and 'pre' not in classes:
trial = args.trial
place = args.place
for images in tqdm(os.listdir(os.path.join(path,'images',trial, place))):
    if 'jpeg' in images and args.aug in images:
        # print(trial, classes,images)
        os.makedirs(os.path.join(path,'labels',trial, place),exist_ok = True)
        os.makedirs(os.path.join(path,'pixel_labels',trial, place),exist_ok = True)
        txt_path = os.path.join(path,'labels',trial,place, '{}.txt'.format(images[:-5]))
        np_path = os.path.join(path,'pixel_labels',trial,place, '{}.npy'.format(images[:-5]))
        # f = open(txt_path,'r').readlines()
        if os.path.isfile(np_path):
            print('you already annotated this image!')
        else:
            print('{}'.format(images))
            cnt=0
            img = cv2.imread(os.path.join(path,'images',trial,place,images))
            a = []
            b = []
            c = []
            
            cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("image", np.array(img).shape[0]*10,np.array(img).shape[1]*10)
            cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
            img = cv2.resize(img, (np.array(img).shape[0]*10, np.array(img).shape[1]*10))                    # Resize image
            cv2.circle(img, (np.array(img).shape[0]//2, np.array(img).shape[1]//2), 5, (0, 255, 0), thickness=-1)
            cv2.imshow("image", img)

            cv2.waitKey(0)
            print(a)
            print(b)
            
            if len(a)>0:
                pix_annotation = []
                for i in range(len(a)):
                    pix_annotation.append(np.array([[int(a[i]/10),int(b[i]/10)]]))
                pix_annotation = np.concatenate(pix_annotation,0)
                print(pix_annotation)

                # np.savetxt(txt_path, yolo_annotation,'%.5f')
                np.save(np_path, pix_annotation)

                print('saved to {}.txt\n\n\n'.format(images))

            else:
                # yolo_annotation = []
                # np.savetxt(txt_path, yolo_annotation,'%.5f')

                pix_annotation = []
                np.save(np_path, pix_annotation)

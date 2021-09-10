import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
from natsort import natsorted


# Picture path

def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(0)
        cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)
        cv2.rectangle(img, (x-50, y-50),(x+50,y+50),  (0, 0, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(1)
        cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=-1)
        cv2.rectangle(img, (x-50, y-50),(x+50,y+50),  (0, 255, 0), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--trial', default=None)

args = parser.parse_args()

path = '../AVM_center_data_track/'

pad = 50
trial = args.trial
cnt=0
while True:
    cnt+=1
    img_cnt =90
    os.makedirs(os.path.join(path, 'simclr'),exist_ok = True)
    for images in tqdm(natsorted(os.listdir(os.path.join(path,'images',trial)))):
#         print
        if 'jpeg' in images:
            txt_path = os.path.join(path,'labels',trial,'{}.txt'.format(images[:-5]))
            np_path = os.path.join(path,'pixel_labels',trial,'{}.npy'.format(images[:-5]))
            # f = open(txt_path,'r').readlines()

            print('{}'.format(images))
            img = cv2.imread(os.path.join(path,'images',trial,images))
            a = []
            b = []
            c = []
            raw_img = img.copy()

            cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("image", 800,800)
            cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
            img = cv2.resize(img, (800, 800))                    # Resize image
            raw_img = cv2.resize(raw_img, (800, 800))                    # Resize image
            cv2.imshow("image", img)

            cv2.waitKey(0)
            print(a)
            print(b)
            
            for i in range(len(a)):
                crop_img = raw_img[b[0]-pad:b[0]+pad, a[0]-pad:a[0]+pad]
                cv2.imwrite(os.path.join(path, 'simclr',"{}.png".format(img_cnt)), crop_img)
                img_cnt +=1


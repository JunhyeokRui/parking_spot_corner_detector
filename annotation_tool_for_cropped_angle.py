import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import copy
from PIL import Image

# Picture path

def on_EVENT_BUTTONDOWN(event, x, y, flags, param):

    this_img = copy.deepcopy(img)
    
    if event == cv2.EVENT_MOUSEMOVE:
        previous_img = this_img
        xy = "%d,%d" % (x, y)
        # cv2.circle(this_img, (x, y), 3, (0, 255, 255), thickness=-1)
        # cv2.putText(this_img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    # 1.0, (0, 0, 0), thickness=1)
        # cv2.line(this_img, (x, y),(np.array(this_img).shape[0]//2,np.array(this_img).shape[1]//2),  (0, 255, 255), 2)
        # this_img = previous_img


    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        # c.append(0)
        # if len(a)==0:
        cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)

        # if len(a)>1:
            # cv2.line(img, (a[-1], b[-1]),(a[-2],b[-2]),  (0, 255, 255), 2)
        # elif len(a)==1:
        cv2.line(img, (x, y),(np.array(this_img).shape[0]//2,np.array(this_img).shape[1]//2),  (0, 255, 0), 2)
        this_img = img

    cv2.imshow("image", this_img)
    # print(x,y)

        # return x, y
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     xy = "%d,%d" % (x, y)
    #     a.append(x)
    #     b.append(y)
    #     c.append(1)
    #     cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=-1)
    #     cv2.rectangle(img, (x-30, y-30),(x+30,y+30),  (0, 255, 0), 2)
    #     cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
    #                 1.0, (0, 0, 0), thickness=1)
    #     cv2.imshow("image", img)
    #     print(x,y)
    #     # return x, y


parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--trial', default=None)
parser.add_argument('--place', default=None)
parser.add_argument('--aug', default=None)

args = parser.parse_args()

path = '/home/dyros/yhpark/new_refinement_data/'

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

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
        os.makedirs(os.path.join(path,'atan2_labels',trial, place),exist_ok = True)
        # os.makedirs(os.path.join(path,'atan_labels',trial, place),exist_ok = True)
        os.makedirs(os.path.join(path,'coord_labels',trial, place),exist_ok = True)

        atan2_path = os.path.join(path,'atan2_labels',trial,place, '{}.npy'.format(images[:-5]))
        # atan_path = os.path.join(path,'atan_labels',trial,place, '{}.npy'.format(images[:-5]))
        coord_path = os.path.join(path,'coord_labels',trial,place, '{}.npy'.format(images[:-5]))

        # f = open(txt_path,'r').readlines()
        if os.path.isfile(atan2_path):
            print('you already annotated this image!')
        else:
            print('{}'.format(images))
            cnt=0
            img = cv2.imread(os.path.join(path,'images',trial,place,images))
            # pil_img = Image.open(os.path.join(path,'images',trial,place,images)).convert('RGB')
            # pil_img = change_contrast(pil_img, 50)

            # cv_img = np.array(pil_img)
            # img = cv_img[:,:,::-1].copy()

            a = []
            b = []
            
            cv2.namedWindow("image")
            cv2.resizeWindow("image", np.array(img).shape[0]*3,np.array(img).shape[1]*3)
            cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
            img = cv2.resize(img, (np.array(img).shape[0]*3, np.array(img).shape[1]*3))                    # Resize image
            cv2.circle(img, (np.array(img).shape[0]//2, np.array(img).shape[1]//2), 5, (0, 255, 0), thickness=-1)
            cv2.imshow("image", img)

            cv2.waitKey(0)
            print(a)
            print(b)
            
            if len(a)==1:
                atan2_annotation = []
                # atan_annotation = []
                coord_annotation = []
                for i in range(len(a)):
                    atan2_annotation.append(np.array([np.arctan2(-b[0]+np.array(img).shape[1]//2, a[0]-np.array(img).shape[0]//2)/np.pi]))
                    # atan_annotation.append(np.array([np.arctan((-b[0]+np.array(img).shape[1]//2)/(a[0]-np.array(img).shape[0]//2))/np.pi]))
                    temp = np.array([-b[0]+np.array(img).shape[1]//2,a[0]-np.array(img).shape[0]//2])
                    coord_annotation.append(temp/np.linalg.norm(temp))

                atan2_annotation = np.concatenate(atan2_annotation,0)
                # atan_annotation = np.concatenate(atan_annotation,0)
                coord_annotation = np.concatenate(coord_annotation,0)

                print(atan2_annotation)
                # print(atan_annotation)
                print(coord_annotation)


                # np.savetxt(txt_path, yolo_annotation,'%.5f')
                np.save(atan2_path, atan2_annotation)
                # np.save(atan_path, atan_annotation)
                np.save(coord_path, coord_annotation)

                print('saved to {}.txt\n\n\n'.format(images))
            
            # elif len(a)>1:
            #     pix_annotation = []
            #     # for i in range(len(a)):
            #     pix_annotation.append(np.array([np.arctan2(-b[-1]+b[-2], a[-1]-a[-2])/np.pi]))
            #     pix_annotation = np.concatenate(pix_annotation,0)
            #     print(pix_annotation)

            #     # np.savetxt(txt_path, yolo_annotation,'%.5f')
            #     np.save(np_path, pix_annotation)

            #     print('saved to {}.txt\n\n\n'.format(images))

            else:
                # yolo_annotation = []
                # np.savetxt(txt_path, yolo_annotation,'%.5f')

                raise('Click Something.')
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
        cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)
        cv2.rectangle(img, (x-30, y-30),(x+30,y+30),  (0, 0, 255), 2)
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

args = parser.parse_args()

path = '/home/dyros/yhpark/AVM_center_data_4class/'
orig_path = '/home/dyros/yhpark/AVM_center_data/'


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
for images in tqdm(os.listdir(os.path.join(path,'images',trial))):
    if 'jpeg' in images:
        # print(trial, classes,images)
        os.makedirs(os.path.join(path,'labels',trial),exist_ok = True)
        os.makedirs(os.path.join(path,'pixel_labels',trial),exist_ok = True)
        txt_path = os.path.join(path,'labels',trial,'{}.txt'.format(images[:-5]))
        orig_txt_path = os.path.join(orig_path,'labels',trial,'{}.txt'.format(images[:-5]))

        np_path = os.path.join(path,'pixel_labels',trial,'{}.npy'.format(images[:-5]))
        # f = open(txt_path,'r').readlines()
            # yolo_annotation = np.loadtxt(orig_txt_path)

        if os.path.isfile(txt_path):
            # yolo_annotation = []
            print('you already annotated this image')

            print('checking sanity')
            yolo_annotation = np.loadtxt(txt_path)
            print(yolo_annotation.shape)
            if not len(yolo_annotation.shape)==2:
                print('not sane. correcting label.')
                yolo_annotation = np.expand_dims(yolo_annotation,0)
                np.savetxt(txt_path, yolo_annotation,'%.5f')

        else:
            yolo_annotation = np.loadtxt(orig_txt_path)
            # pix_annotation = []
            pix_annotation = np.load(np_path)

            print('{}'.format(images))
            cnt=0
            img = cv2.imread(os.path.join(path,'images',trial,images))
            a = []
            b = []
            c = []
            
            cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("image", 1200,1200)
            cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
            img = cv2.resize(img, (1200, 1200))                    # Resize image
            cv2.imshow("image", img)
            print(yolo_annotation)
            cv2.waitKey(0)
            # print(a)
            # print(b)
            print(len(a))
            if len(a)>0:
                for i in range(len(a)):
                    if len(yolo_annotation.shape)==2:
                        if c[i]==0:
                            yolo_annotation = np.concatenate((yolo_annotation,np.array([[int(2),a[i]/1200,b[i]/1200,30/1200,30/1200]])))
                        elif c[i]==1:
                            yolo_annotation = np.concatenate((yolo_annotation,np.array([[int(3),a[i]/1200,b[i]/1200,30/1200,30/1200]])))
                    else:
                        yolo_annotation = np.expand_dims(yolo_annotation,0)
                        if c[i]==0:
                            yolo_annotation = np.concatenate((yolo_annotation,np.array([[int(2),a[i]/1200,b[i]/1200,30/1200,30/1200]])))
                        elif c[i]==1:
                            yolo_annotation = np.concatenate((yolo_annotation,np.array([[int(3),a[i]/1200,b[i]/1200,30/1200,30/1200]])))

                # yolo_annotation = np.concatenate(yolo_annotation,0)
                print(yolo_annotation) 

                # pix_annotation = []
                # for i in range(len(a)):
                #     pix_annotation.append(np.array([[c[i], int(a[i]/3),int(b[i]/3)]]))
                # pix_annotation = np.concatenate(pix_annotation,0)
                # print(pix_annotation)

                np.savetxt(txt_path, yolo_annotation,'%.5f')
                # np.save(np_path, pix_annotation)

                print('saved to {}.txt\n\n\n'.format(images))

            # else:
            #     # yolo_annotation = []
            #     np.savetxt(txt_path, yolo_annotation,'%.5f')

            #     # pix_annotation = []
            #     # np.save(np_path, pix_annotation)
            else:
                np.savetxt(txt_path, yolo_annotation,'%.5f')

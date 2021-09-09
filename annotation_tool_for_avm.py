import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--trial', required=True)
parser.add_argument('--dataset', default='AVM_center_data_4class_test')
parser.add_argument('--img_size', default=600, type=int)
parser.add_argument('--bb_size', default=30,type=int)

args = parser.parse_args()

bbsize = args.bb_size

def on_EVENT_BUTTONDOWN_a(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(0)
        cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)
        cv2.rectangle(img, (x-bbsize, y-bbsize),(x+bbsize,y+bbsize),  (0, 0, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)

def on_EVENT_BUTTONDOWN_s(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(1)
        cv2.circle(img, (x, y), 3, (0, 255, 255), thickness=-1)
        cv2.rectangle(img, (x-bbsize, y-bbsize),(x+bbsize,y+bbsize),  (0, 255, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)

def on_EVENT_BUTTONDOWN_d(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(2)
        cv2.circle(img, (x, y), 3, (255, 0, 255), thickness=-1)
        cv2.rectangle(img, (x-bbsize, y-bbsize),(x+bbsize,y+bbsize),  (255, 0, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)

def on_EVENT_BUTTONDOWN_f(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(3)
        cv2.circle(img, (x, y), 3, (180, 0, 0), thickness=-1)
        cv2.rectangle(img, (x-bbsize, y-bbsize),(x+bbsize,y+bbsize),  (180, 0, 0), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)


path = '../{}/'.format(args.dataset)

trial = args.trial

key = None
for images in tqdm(os.listdir(os.path.join(path,'images',trial))):
    if 'jpeg' in images:
        # print(trial, classes,images)
        os.makedirs(os.path.join(path,'labels',trial),exist_ok = True)
        txt_path = os.path.join(path,'labels',trial,'{}.txt'.format(images[:-5]))
        # f = open(txt_path,'r').readlines()
        if os.path.isfile(txt_path):
            # print('you already annotated this image!')
            # yolo_annotation = []
            # yolo_annotation = np.loadtxt(txt_path)
            pass
        else:
            # pix_annotation = []
            # pix_annotation = np.load(np_path)

            print('\n{}'.format(images))
            print("Specify the type of marking point you want to annotate: [A/S/D/F] = \n A: Out, S: In, D: Out_aux, F: In_aux")

            cnt=0
            img = cv2.imread(os.path.join(path,'images',trial,images))
            img = cv2.resize(img, (args.img_size, args.img_size))                    # Resize image
            a = []
            b = []
            c = []
            

            while True:
                cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("image", args.img_size,args.img_size)
                cv2.imshow("image", img)
                key = cv2.waitKey()
                # if
                if key==ord("x"):
                    print(a)
                    print(b)
                    
                    if len(a)>0:
                        yolo_annotation = []
                        for i in range(len(a)):
                            if c[i]==0:
                                yolo_annotation.append(np.array([[int(0),a[i]/args.img_size,b[i]/args.img_size,30/args.img_size,30/args.img_size]]))
                            elif c[i]==1:
                                yolo_annotation.append(np.array([[int(1),a[i]/args.img_size,b[i]/args.img_size,30/args.img_size,30/args.img_size]]))
                        yolo_annotation = np.concatenate(yolo_annotation,0)
                        print(yolo_annotation)

                        np.savetxt(txt_path, yolo_annotation,'%.5f')
                        # np.save(np_path, pix_annotation)

                        print('saved to {}.txt\n\n\n'.format(images[:-5]))


                    else:
                        yolo_annotation = []
                        np.savetxt(txt_path, yolo_annotation,'%.5f')

                        # pix_annotation = []
                        # np.save(np_path, pix_annotation)
                        
                    break

                if key==ord("a"):
                    print("\n** A: Marking Outside Corner Points")
                    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN_a)
                if key==ord("s"):
                    print("\n** S: Marking Inside Corner Points")
                    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN_s)
                if key==ord("d"):
                    print("\n** D: Marking Outside Auxiliary Marking Points")
                    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN_d)
                if key==ord("f"):
                    print("\n** F: Marking Inside Auxiliary Marking Points")
                    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN_f)

                if key==ord("p"):
                    print("Passing this image. You have to annotated again!\n\n")
                    break
                if key==ord("q"):
                    print("Quitting.")
                    break

    if key==ord("q"):
        break

# [DYROS x AIIS] PSDT-Net: Parking Slot Detection and Tracking Algorithm


## System Overview
![Overview](https://cln.sh/SZexiN/download)

This algorithm detects visibles corner points in parking spots on AVM inputs, and estimates the parking spot position and orientation based on using the detected corner point information. Main code of YOLO training and detecting is developed based on existing YOLOv5 code ([https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)), and  position estimator is developed from scratch. 

## Technical Details

### Conda Environment

I used conda environment to take care of the dependencies of multiple packages and libraries. Anaconda is similar to Docker in a sense that it provides isolated fully-configurable environments allowing the developers to avoid dependency issues. If you're not familiar with anaconda, I recommend reading this [article](https://medium.com/pankajmathur/what-is-anaconda-and-why-should-i-bother-about-it-4744915bf3e6).

I've created anaconda environment named ```yhpark``` with python version 3.7 for all experiments regarding this project. It is equipped with the important libraries such as PyTorch, OpenCV and else. You can activate the conda environment using

```
conda activate yhpark
```

For DeepSORT, I created a separate environment named ```deepsort```. For any kind of DeepSORT related experiments, you must activate this environment. 

```
conda activate deepsort
```

**If you want to create your own conda environment on a new computer**, follow the steps below: 

1. Install Anaconda. Remember to restart your terminal after installing.
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
    chmod +x ./Anaconda3-2021.05-Linux-x86_64.sh
    ./Anaconda3-2021.05-Linux-x86_64.sh
    ``` 
2. Create new conda environment with a specific python version. Any python version over 3.6+ would be fine for modern deep learning experiments. 
    ```
    conda create -n YOUR_ENVIRONMENT_NAME python=3.7
    ```
3. Install all the python packages I used using ```requirements.txt``` file. 
   ```
   pip install -r requirements.txt
   ```
   **Disclaimer:** You may have to install different version of PyTorch depending on your GPU's CUDA version. I used CUDA version 10.1 for this project, but it doesn't matter. If you have CUDA 11.0+, no worries! Just reinstall PyTorch with CUDA 11.1. Installing PyTorch is really easy: [https://pytorch.org](https://pytorch.org).



### Accessing the computer I used via SSH

For testing, you can use the exact same computer I used for this project. I recommend using it remotely, since dealing with two or more physical computers in front of you can be quite burdensome. 

If you're connected to the local network of the lab, you can access the computer I used for experiments via SSH. Run the following command in your own terminal. 

```
ssh dyros@192.168.0.47
```

If you're accessing from external network outside of the lab, you can access vai port 1001: I've forwarded the port 1001 to SSH port 22 for local IP. 

```
ssh dyros@147.46.19.190 -p 1001
```

**Disclaimer:** I've been using Google Chrome Remote Desktop for screen sharing, but it allows only one Google account to access the computer. I can transfer the Google account ownership if needed. Otherwise, you may configure AnyDesk or Microsoft's Remote Desktop client RDP in the future. 


## Dataset
Image-type dataset is first generated from raw bagfiles recorded during actual parking experiments. (Run ``rosrun yhpark_psd image_saver.py`` to save imagefiles from existing bagfiles.) Total 6 parking experiments on different parking spots are executed, and we name those experiments as trial A~F. Network training is usually done with trial A,B,C and testing is remaining trials are used for testing. 

Datasets is (and should be) stored right outside this repository, using following folder structure. 

```
..
├── AVM_center_data_4class             
│   ├── Images
│   │   ├── trial_A
│   │   │   ├── image_001.jpeg
│   │   │   ├── image_002.jpeg
│   │   │   └── ...
│   │   ├── trial_B
│   │   ├── trial_C
│   │   └── ...
│   ├── Labels
│   │   ├── trial_A
│   │   │   ├── image_001.txt
│   │   │   ├── image_002.txt
│   │   │   └── ...
│   │   ├── trial_B
│   │   ├── trial_C
│   │   └── ...
└── └── README.md

```
``../AVM_center_400/Labels/trial_*/image_*.txt`` is the corresponding YOLO label for the image  ``../AVM_center_400/Images/trial_*/image_*.jpeg``.


## Marking Point Detection

### Overview

This system uses YOLO to precisely estimate the position of each corner points. You might wonder, how can YOLO, basically an object detection algorithm, be used for such precise point localization task. Surprisingly, without any additional help of cascading techniques, YOLO alone itself was able to precisely estimate the position of such keypoints. This allowed much faster detection time compared to other techniques. 

To see the details of YOLO architecture and understand how and why it works, I recommend reading the following article: 



### Annotation 

Checkout the [YOLOv5 code repository](https://github.com/ultralytics/yolov5) to see how YOLO label is typically formatted. Following python code is written to generate labels following the format that YOLOv5 requires for training. 

**Disclaimer:** Due to limitation of OpenCV's cursor-related function, only two classes can be differentiated per one execution: Left-click and Right-click. But I believe there's a better and clever solution that can mark all 4 classes at once using some kind of keyboard inputs alongside with mouse clicks! 

```
class 0 : inner-side corner
class 1 : outer-side corner
class 2 : inner-side auxiliary point
class 3 : outer-side auxiliary point
```

Thus, to create label with 4 classes, I had to go through the dataset two times. Again, you can do this in a better way. 

```
python annotation_tool_for_avm.py --trial trial_D --class 0,1 ## label class 0,1
python annotation_tool_for_avm.py --trial trial_D --class 2,3 ## label class 2,3
```

There are other arguments that can be adjusted, including the size of the bounding box. To see all the adjustable arguments, run:

```
python annotation_tool_for_avm.py --help
```

### Training

To create train / test / validation split based on parking episodes, you must run the following code. 

```
python split_generator.py --train A,B,C --valid D,E --test F --dataset AVM_center_data --run split_ABC_DE_F.yaml
```

This code automatically creates ```./data/split_ABC_DE_F.yaml``` file that is required for YOLO training. You can specify this split by passing the YAML file name through python argument. 

```
python train.py --batch-size 128 --data data/split_ABC_DE_F.yaml --name EXPERIMENT_NAME_OF_YOUR_CHOICE
```

Running this creates a YOLO training weight under ```./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt```. 

YOLO, like any other deep learning models, requires large amount of data, and large batch size. In that sense, RTX2060 Super with 8GB VRAM was not sufficient enough for YOLO training. I used my personal deep learning GPU server equipped with Tesla-T4 with 15GB VRAM and RTX3080TI with 12GB VRAM. If retraining is required in some time in the future, I recommend using RTX3090 with 24GB VRAM (if possible) with maximum batchsize that VRAM allows. Fortunately, YOLOv5 code is quite intuitively written and highly customizable. If enough GPU power is ready, training will not become a problem. 


### Check the Training result

YOu might want to check the training result via running detections on image-level. You can use the following code to check out the detection. Detection runs per trial. 

```
python detect.py --weights ./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt --source ../AVM_center_data_4class/image/trial_F --name test_trial_F
```

This code saves the detection result on ```./runs/detect/test_trial_F/```. 

### Detection on ROS

I don't know if this is a preferable solution or not in ROS community, but I've written a single python file that publishes the YOLO detection result in a concatenated string format. 

```
python ROS_detect_marking_points.py --yolo_weight ./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt --view-img
```

### Marking Point Detection WorkFlow


https://user-images.githubusercontent.com/68195716/132668170-fafce57b-352e-41f5-8c2d-db991a7137a4.mp4



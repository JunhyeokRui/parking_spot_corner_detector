
# [DYROS x AIIS] PSDT-Net: Parking Slot Detection and Tracking Algorithm


## System Overview
<!-- <center> -->
<!-- ![Overview](https://cln.sh/SZexiN/download) -->
<center>
<img src="https://cln.sh/SZexiN/download" width="70%">
</center>

This algorithm detects visibles corner points in parking spots on AVM inputs, and estimates the parking spot position and orientation based on using the detected corner point information. Main code of YOLO training and detecting is developed based on existing YOLOv5 code ([https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)), and  position estimator is developed from scratch. 

## Technical Details

### Conda Environment

I used conda environment to take care of the dependencies of multiple packages and libraries. Anaconda is similar to Docker in a sense that it provides isolated fully-configurable environments allowing the developers to avoid dependency issues. If you're not familiar with anaconda, I recommend reading this [article](https://medium.com/pankajmathur/what-is-anaconda-and-why-should-i-bother-about-it-4744915bf3e6).

I've created anaconda environment named ```yhpark``` with python version 3.7 for all experiments regarding this project. It is equipped with the important libraries such as PyTorch, OpenCV and else. You can activate the conda environment using

```
conda activate yhpark
```
<!-- 
For DeepSORT, I created a separate environment named ```deepsort```. For any kind of DeepSORT related experiments, you must activate this environment. 

```
conda activate deepsort
``` -->

**If you want to create your own conda environment on a new computer**, follow the steps below: 

1. Install Anaconda. Remember to restart your terminal (or run ```source ~/.bashrc``` if you use ```bash```, ```source ~/.zshrc``` if you use ```zsh```) after installing.
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
    chmod +x ./Anaconda3-2021.05-Linux-x86_64.sh
    ./Anaconda3-2021.05-Linux-x86_64.sh
    ``` 
2. Create new conda environment with a specific python version. Any python version over 3.6+ would be fine for modern deep learning experiments. 
    ```
    conda create -n YOUR_ENVIRONMENT_NAME python=3.7
    ```
3. Install all the python packages I used using ```requirements_pytorch.txt``` file. 
   ```
   pip install -r requirements_pytorch.txt
   ```
   **Disclaimer:** I deliberately deleted the PyTorch-related python packages from above textfile, since you may have to install different pytorch pacakges depending on your CUDA version. **Install PyTorch from [https://pytorch.org](https://pytorch.org) using pip**. 



### Accessing the computer I used via SSH

For testing, you can use the exact same computer I used for this project. I recommend using it remotely, since dealing with two or more physical computers in front of you can be quite burdensome. 

If you're connected to the local network of the lab, you can access the computer I used for experiments via SSH. Run the following command in your own terminal. 

```
ssh dyros@192.168.0.47
```

If you're accessing from external network outside of the lab, you can access via port 1001: I've forwarded the port 1001 to SSH port 22 for my laptop. 

```
ssh dyros@147.46.19.190 -p 1001
```

**Disclaimer:** I've been using Google Chrome Remote Desktop for screen sharing, but it allows only one Google account to access the computer. I can transfer the Google account ownership if needed. Otherwise, you may configure AnyDesk or Microsoft's Remote Desktop client RDP in the future. 


## Dataset
Image-type dataset is first generated from raw bagfiles recorded during actual parking experiments. (Run ``rosrun yhpark_psd image_saver.py`` to save imagefiles from existing bagfiles.) Total 6 parking experiments on different parking spots are executed, and we name those experiments as trial A~F. Network training is usually done with trial A,B,C and remaining trials are used for testing. 

Dataset is (and should be) stored right outside this repository, using following folder structure. 

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

To see the details of YOLO architecture and understand how and why it works, I recommend reading the following article: [YOLO Explained. What is YOLO?](https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31)



### Annotation 

Checkout the [YOLOv5 code repository](https://github.com/ultralytics/yolov5) to see how YOLO label is typically formatted. Following python code is written to generate labels following the format that YOLOv5 requires for training. 

<!-- **Disclaimer:** Due to limitation of OpenCV's cursor-related function, only two classes can be differentiated per one execution: Left-click and Right-click. But I believe there's a better and clever solution that can mark all 4 classes at once using some kind of keyboard inputs alongside with mouse clicks!  -->
```
python annotation_tool_for_avm.py --trial trial_D --img_size 500 --bb_size 20
```
Before clicking on the marking point, you should specify the type of the point by keyboard input. 

```
class 0 : outer-side corner  (Key = A)
class 1 : inner-side corner  (Key = S)
class 2 : outer-side auxiliary point  (Key = D)
class 3 : outer-side auxiliary point  (Key = F)
```

Checkout the demo to see how this annotation tool works. Pay attention to the keyboard inputs. 



https://user-images.githubusercontent.com/68195716/132684641-eb6b5b43-c573-4825-8aac-03e874531b3c.mp4




### Training

To create train / test / validation split based on parking episodes, you must run the following code. 

```
python split_generator.py --train A,B,C --valid D,E --test F --dataset AVM_center_data --run split_ABC_DE_F.yaml
```

This code automatically creates ```./data/split_ABC_DE_F.yaml``` file that is required for YOLO training. You can pass this split YAML file through python argument during training. 

```
python train.py --batch-size 128 --data data/split_ABC_DE_F.yaml --name EXPERIMENT_NAME_OF_YOUR_CHOICE
```

Running this saves a trained weight under ```./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt```. 

**Note:** YOLO, like any other deep learning models, requires large amount of data, and large batch size. In that sense, RTX2060 Super with 8GB VRAM is not optimal for YOLO training (although it can technically work). I used my personal deep learning GPU servers each equipped with Tesla-T4 (15GB VRAM)and RTX3080TI (12GB VRAM). If retraining is required in some time in the future, I recommend using RTX3090 computer with 24GB VRAM (if possible) with maximum batchsize that the GPU allows. Fortunately, YOLOv5 code is quite intuitively written and is highly customizable. If enough GPU power is ready, training will not be a problem. I've made some changes to the code that better suits the point detection task. 


### Check the Training result

Before running on ROS, you might want to check the training result on image files. You can use the following code to check out the detection results. Detection code runs per trial. 

```
python detect.py --weights ./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt --source ../AVM_center_data_4class/image/trial_F --name test_trial_F
```

This code saves the detection result on ```./runs/detect/test_trial_F/```. 

### Detection on ROS

I've written a simple python code that publishes the YOLO detection result in a concatenated string format. This concatenated string format will be properly subscribed by ```ROS_inference.py``` code, explained below. 

```
python ROS_detect_marking_points.py --yolo_weight ./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt --view-img
```

You should use ```--view-img``` flag to open up an opencv window that shows the live detection results. 

### Marking Point Detection WorkFlow


https://user-images.githubusercontent.com/68195716/132668170-fafce57b-352e-41f5-8c2d-db991a7137a4.mp4



## Parking Slot Inference

Using the detected marking points, PSDT-Net infers the position of parking slots. You can use 

```
python ROS_inference.py --view-img
```
to see its results in real-time. Remember, you should run  ```ROS_detect_marking_points.py``` before you run the ```ROS_inference.py``` script. 



## Marking Point Tracking

### Overview

PSDT-Net uses DeepSORT to track different marking points. DeepSORT is a deep learning variant of SORT algorithm. SORT performs data association using just the bounding box size and location. However, this can be problematic in our case where the bounding box size is quite static. DeepSORT can be a solution here, since it actually **sees** the content inside the bounding box using CNN. I highly recommend reading this article if you're not familiar with DeepSORT: [DeepSORT Explained](https://nanonets.com/blog/object-tracking-deepsort/). 

### Run with pretrained weight

Even with a pretrained feature extractor network, DeepSORT can quite robustly track multiple marking points. You can check the result with pretrained weights using the code below:

```
python ROS_track_marking_points.py --view-img
```

### Create dataset for feature extractor

But as you can see, pretrained weight trained on Market1501 dataset (suited for human tracking) induces a lot of label switching. It might be appropriate to train a custom feature extractor network to increase the robustness of DeepSORT. What we should first do is to **create a new dataset** for tracking. The dataset should contain folders of diferent marking points, captured from different instances. The dataset has the following structure. 

```
..
├── dyros_deepsort_dataset           
│   ├── cropped_30
│   │   ├── point_1
│   │   │   ├── point_1_001.jpeg
│   │   │   ├── point_1_002.jpeg
│   │   │   ├── point_1_003.jpeg
│   │   │   └── ...
│   │   ├── point_2
│   │   │   ├── point_2_001.jpeg
│   │   │   ├── point_2_002.jpeg
│   │   │   ├── point_2_003.jpeg
│   │   │   └── ...
│   │   ├── point_3
│   │   ├── point_4
│   │   ├── ...
│   │   └── ...
│   ├── pixel_labels
│   │   ├── point_1
│   │   │   ├── point_1_001.txt
│   │   │   ├── point_1_002.txt
│   │   │   ├── point_1_003.txt
│   │   │   └── ...
│   │   ├── point_2
│   │   │   ├── point_2_001.txt
│   │   │   ├── point_2_002.txt
│   │   │   ├── point_2_003.txt
│   │   │   └── ...
│   │   ├── point_3
│   │   ├── point_4
│   │   ├── ...
│   │   └── ...
└── └── README.md
```

Note, ```point_a``` and ```point_b``` are physically differernt marking points, while ```point_a_00x.jpeg``` and ```point_a_00y.jpeg``` are the same physical marking points, captured from different frames. I created this dataset using the following annotation tool:

```
python annotation_tool_for_avm_deepsort.py --trial trial_A --crop 20
```

If you want to generate a new cropped dataset with different ```bb_size```, use:

```
python annotation_tool_for_avm_deepsort.py --crop-only --trial trial_A --crop 40 
```


### Training siamese network with Triplet loss

I used the famous [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library to train the feature extractor network.  Check out ```siaemese_training.py``` code that I used for training. It uses ResNet as a CNN architecture. 

```
python siamese_training.py --crop-size 30 --margin 0.3 --epoch 300 --name TEST
```

Running this code saves its trained weight in the following directory: ```./deep_sort_pytorch/deep_sort/deep/TEST_last.pth```.

You can perform hyperparameter tuning adjusting the following arguments; 

```
python siamese_training.py --help

>  --crop-size # bounding box size around the marking point, default = 30
>  --margin # minimum distance between the embeddings of different marking points, default = 0.3
>  --batch-size  # default = 64
>  --rot_aug # default = True
>  --gaussian_aug # default = False
>  --resize # default = 0 (no resizing)
```


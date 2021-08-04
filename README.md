
## [DYROS x Phantom x AIIS] Parking Spot Position & Orientation Estimation Algorithm


### System Overview
![Overview](https://cln.sh/SZexiN/download)

This algorithm detects visibles corner points in parking spots on AVM inputs, and estimates the parking spot position and orientation based on the corner point inputs. Main code of YOLO training and detecting is developed based on existing YOLOv5 code ([https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)), and  position estimator is developed from scratch. All codes are developed for ROS-Melodic system. 

### YOLO Detection

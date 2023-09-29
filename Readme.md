# Camera - Lidar Sensor Fusion and Segmentation of point cloud using deeplabv3
Sreehari Premkumar, Northeastern University, MS ROBOTICS

OS : Ubuntu 20.04 LTS
with VSCode and Python 3.10 Pytoech Torchvision

Title : Semantic Segmentation using Neural Nets + 3D Lidar Sensor Fusion

Objective: The objective of this project is to develop a system that can accurately identify
and track objects in a driving environment using a combination of neural networks,
semantic segmentation, and 3D LiDAR fusion.

Dataset : KITTI Dataset with Camera Images and 3D lidar data

Methodology :
● A pretrained neural network will be used on the collected dataset to perform semantic
segmentation on new video frames captured by the camera in the vehicle.
● The trained model will be able to differentiate between different object classes and
produce a semantic segmentation mask for each frame.
● Fusing camera and lidar point using Spherical Projection to project 3d points onto
Image Frame and camera-lidar extrinsic calibration.
● Then segmenting corresponding points of the point cloud based on camera image
segmentation.


Link to Report:
https://drive.google.com/file/d/1I8qeuT5yFvU-kK82aeNoJupUS8OElcug/view?usp=sharing

[Original Input Video](https://drive.google.com/file/d/1jCaJz7pN9qUhqiCtx2KNXgbFFuqf5HIa/view?usp=sharing)
[Point Cloud Data](https://drive.google.com/file/d/1OZMlq8hGZTL6rB26DwQgXkk1LNps2OSh/view?usp=sharing)
[Segmentation Output](https://drive.google.com/file/d/1kBOoeGkaspR_NpWTD3AxCkzfBeuKvtJL/view?usp=sharing)
[Point Cloud Lidar data fused with camera](https://drive.google.com/file/d/1sIGKNei4ddMTZVhBiFqDcmDX_lH1IDne/view?usp=sharing)

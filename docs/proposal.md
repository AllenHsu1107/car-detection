# Project Proposal
## 1. Motivation & Objective

Ensuring a safe distance between vehicles while driving is critical to prevent collisions. In California, this guideline, known as the "3-second rule," dictates that the gap between your vehicle and the one in front should exceed the distance your current speed would cover in 3 seconds.

Our objective is to create a driving assistance system that monitors the current speed of both your car and the one in front, while also gauging the distance separating the two vehicles. When this distance falls short of the 3-second rule, our system alerts the driver.

Moreover, we aim to extend this concept to the vehicle behind as well. This entails detecting the distance and speed of the car trailing behind. If this distance is insufficient, a warning signal will be triggered to alert the trailing vehicle.

## 2. State of the Art & Its Limitations

Presently, distance detection is commonly accomplished through radar technology, which can encounter challenges on curved roads where radar signals might have difficulty reaching the vehicle ahead effectively.

## 3. Novelty & Rationale

Instead of relying on radar, our approach involves utilizing cameras to identify the car in front and compute the distance. We've devised several methods to achieve this:

**License Plate Detection**: Identifying the license plate of the lead vehicle and using the size ratio between its appearance in the camera and its actual dimensions to calculate the distance.

**Dual Camera Setup**: Employing two cameras capturing different angles to calculate depth by integrating the data from both sources.

**Vehicle and Lane Appearance Analysis**: Recognizing the vehicle itself and determining the vanishing point of the lanes to calculate the vehicle's relative position.

Using cameras offers the advantage of determining the car's position even when it's not centrally positioned, particularly on curved roads.

## 4. Potential Impact

Current radar systems, while effective in measuring distances and speeds, often face limitations in precise object identification and struggle in scenarios with high-density traffic or sharp curves. In contrast, the use of computer vision in the driving assistance system offers a novel approach that addresses some of these limitations. Therefore, if computer vision is successfully implemented, it will increase the accuracy of car detection, as well as enhance road safety.

## 5. Challenges

There are several challenges associated with utilizing computer vision for this purpose:

**Resolution Limitations**: When the vehicle ahead is at a significant distance, the camera's resolution may cause the license plate to appear small, making detection difficult and resulting in substantial errors. Mitigation could involve employing higher resolution cameras or pausing the process when the distance is too far to yield accurate readings.

**Multiple License Plates**: Heavy traffic scenarios might present multiple visible license plates, complicating the identification of the correct plate of the lead vehicle.

**Lane Switching Issues**: Instances of vehicles changing lanes, either ahead or behind, or another vehicle cutting in between, pose a challenge. Such sudden alterations in distance can cause the system to miscalculate speeds, leading to incorrect warnings.

**Lack of Front License Plates**: In states where front license plates are not mandatory, traditional plate detection methods won't apply. Using two cameras to calculate depth or assessing the entire car's appearance and determining distance based on its width might serve as alternative solutions.

**Reliability of Detected Distances**: The system might encounter inaccuracies in the distances detected. Implementing a mechanism, like the Kalman filter, becomes necessary to discern and discard potentially erroneous distance readings.

## 6. Requirements for Success

To complete car detection, the most substantial skill is to be able to utilize computer vision and machine learning, and manipulate existing tool kits such as OpenCV, Sci-kit learn, etc.

## 7. Metrics of Success

For distance estimation, critical metrics encompass the speed of processing, precision in identifying objects or individuals, and system reliability under diverse environmental conditions. As for the Kalman filter, the primary metric for success revolves around enhancing the accuracy of detected distances.

## 8. Execution Plan

### Yi-Lin Tsai:

Research and implement the Kalman filter to effectively eliminate incorrectly detected distances. This involves developing and integrating the Kalman filter into the system to enhance the accuracy and reliability of the detected distances.

### Jian-Ting Ko:

Focused on estimating the distance and relative velocity between our car and the car in the front. This involves exploring two possible methods:

**License Plate Ratio Computation**: Develop a method to compute the distance and relative velocity by assessing the ratio of the license plate's appearance in the video feed compared to its actual size.

**Dual-Camera Depth Estimation**: Implement a system utilizing dual cameras to estimate distance and relative velocity through depth estimation techniques.

### Yao-Ting Hsu:

Tasked with estimating the distance and relative velocity between our car and the car behind. This involves exploring two potential methods:

**Car Body Detection**: Develop a system to detect the car behind and calculate distance and relative velocity based on the actual car width and its size as captured by the camera.

**Lane Detection**: Implement a system that uses lane detection to calculate the distance and relative velocity from the angle and position of the car in the back relative to our vehicle.

## 9. Related Work
### 9.a. Papers
License Plate Detection and Recognition in Unconstrained Scenarios [1]  
This paper elucidates the methodology for vehicle and license plate detection, crucial for the project's implementation in determining distances.

An introduction to the Kalman filter [2]  
This paper provides a description, a derivation, and a simple example of the Kalman filter.

Depth estimation by dual-camera [3]  
This paper outlines a method to estimate depth using a dual-camera system.

Vision-based Vehicle Detection and Distance Estimation [4]  
This paper describes a approach to estimate the distance of a vehicle by detecting the vehicle itself and the vanish point of the lanes.

### 9.b. Datasets
Car License Plate [5]  
This dataset contains 433 images with bounding box annotations of the car license plates within the image.

Pre-trained License Plate Detection Model [6]

### 9.c. Software
List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

## 10. References
[1] Silva, S.M., Jung, C.R. (2018). License Plate Detection and Recognition in Unconstrained Scenarios. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11216. Springer, Cham. https://doi.org/10.1007/978-3-030-01258-8_36 

[2] Welch, Greg, and Gary Bishop. "An introduction to the Kalman filter." (1995): 2. https://perso.crans.org/club-krobot/doc/kalman.pdf

[3] Zhang, Y., Wadhwa, N., Orts, S., Häne, C., Fanello, S., & Garg, R. (2020). Du2Net: Learning Depth Estimation from Dual-Cameras and Dual-Pixels. European Conference on Computer Vision.
https://arxiv.org/abs/2003.14299 

[4] Donghao Qiao, Farhana H. Zulkernine (2020). "Vision-based Vehicle Detection and Distance Estimation". Conference: 2020 IEEE Symposium Series on Computational Intelligence (SSCI). 

[5] https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data 

[6] https://github.com/quangnhat185/Plate_detect_and_recognize 

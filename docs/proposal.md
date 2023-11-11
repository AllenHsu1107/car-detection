# Project Proposal
## 1. Motivation & Objective

It is important to maintain a safe distance between cars while driving to avoid coliision. In California, this distance is often called the "3-second rule", which means the distance between your vehicle and the vehicle in front should exceed the travel distance of your current speed in 3 seconds.

We aim to make a driving assistance which detected the current speed of your car and the car in front, as well as the distance between the two vehicles. If the distance violates the 3-second rule, we warn the driver.

Additionally, we also make use of this concept to the car in the back, which means the device will also detect the distance and the speed of the car behind, and flashing a warning sign to the back car is the distance is too short.

## 2. State of the Art & Its Limitations

Today, distance detection is often achieved by radar, which may possibly cause problems while driving on curve roads, where radar signals are less likely to successfully reanch the car in front.

## 3. Novelty & Rationale

Instead of radar, we use camera to detect the car in front, and calculate the distance. There are two ways to achieve the goal:

1. Detect the license plate of the car in front, and calculate the distance by the ratio of the plate appearing in the camera and the actual size.

2. Use two camera shooting in different angle, and calculate the depth by combining the two footages.

By using cameras, we can get the position of the car even if the car is not in the middle on curve roads.

## 4. Potential Impact

If the detection is successful, it will make car detection more accurate. (ADD MORE)

## 5. Challenges

There might be some challenges by using computer vision.

First, due to the resolution limit of the camera, if the vehicle in front is too far away, the license plate will become significantly small on the camera, which makes it harder to detect, and ends up in large error. This may be resolved by using higher resolution camera, or simply pause the process if the distance is too far to be considered.

Second, there might be multiple lisence plated showing on the camera, especially if the traffic is heavy. Detecting the correct plate of the car in front is also a foreseeable issue.

Third, lane switching is also a challenge, including the car in front/in the back switching lane, or another car cutting in between. The sudden change of the distance may cause the device to calculate the wrong speed, therefore resulting in incorrect alarms.

Fourth, in some states, cars are not required to have a lisence plate at the front end, so plate detection won't work in these cases. We will have to use two cameras to calculate the depth instead, or detect the appearance of the full car, and calculate the distance by using the width of the car.

Fifth, the detected distances are not always reliable. The system needs to know which detected distances are likely to be errors and discards them. The problem can be solved by using the Kalman filter.

## 6. Requirements for Success

To complete car detection, the most substantial skill is to be able to utilize computer vision and machine learning, and manipulate existing tool kits such as OpenCV, Sci-kit learn, etc.

## 7. Metrics of Success
What are metrics by which you would check for success?
For distance estimating, The metrics include speed of processing, accuracy in identifying objects or people, and system reliability across various environmental conditions.
For the Kalman filter, the metric of success is the improvement of the accuracy of the detected distances.

## 8. Execution Plan
Describe the key tasks in executing your project, and in case of team project describe how will you partition the tasks.

Yi-Lin Tsai: Research and implement Kalman filter to discard the wrong detected distances.
Jian-Ting Ko: Estimate the distance and relative velocity between our car and the front car. The following are two possible ways:
Computing the ratio of the license plate in the video and the real one.
Using depth estimation by dual-camera.

## 9. Related Work
### 9.a. Papers
License Plate Detection and Recognition in Unconstrained Scenarios [1]
This paper elucidates the methodology for vehicle and license plate detection, crucial for the project's implementation in determining distances.

An introduction to the Kalman filter [2]
The paper provides a description, a derivation, and a simple example of the Kalman filter.

Depth estimation by dual-camera [3]
The paper outlines a method to estimate depth using a dual-camera system.


### 9.b. Datasets
Car License Plate [4]
This dataset contains 433 images with bounding box annotations of the car license plates within the image.
Pre-trained License Plate Detection Model [5]

### 9.c. Software
List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

## 10. References
[1] Silva, S.M., Jung, C.R. (2018). License Plate Detection and Recognition in Unconstrained Scenarios. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11216. Springer, Cham. https://doi.org/10.1007/978-3-030-01258-8_36 

[2] Welch, Greg, and Gary Bishop. "An introduction to the Kalman filter." (1995): 2. https://perso.crans.org/club-krobot/doc/kalman.pdf

[3] Zhang, Y., Wadhwa, N., Orts, S., Häne, C., Fanello, S., & Garg, R. (2020). Du2Net: Learning Depth Estimation from Dual-Cameras and Dual-Pixels. European Conference on Computer Vision.
https://arxiv.org/abs/2003.14299 

[4] https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data 
[5] https://github.com/quangnhat185/Plate_detect_and_recognize 

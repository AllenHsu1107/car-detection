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

If the detection is successful, it will make car detection more accurate.

## 5. Challenges

There might be some challenges by using computer vision.

First, due to the resolution limit of the camera, if the vehicle in front is too far away, the lisence plate will become significantly small on the camera, which makes it harder to detect, and ends up in large error. This may be resolved by using higher resolution camera, or simply pause the process if the distance is too far to be considered.

Second, there might be multiple lisence plated showing on the camera, especially if the traffic is heavy. Detecting the correct plate of the car in front is also a foreseeable issue.

Third, lane switching is also a challenge, including the car in front/in the back switching lane, or another car cutting in between. The sudden change of the distance may cause the device to calculate the wrong speed, therefore resulting in incorrect alarms.

Fourth, in some states, cars are not required to have a lisence plate at the front end, so the detection won't work in these cases. We will have to detect the appearance of the full car instead, and calculate the distance by using the width of the car.

## 6. Requirements for Success
What skills and resources are necessary to perform the project?

## 7. Metrics of Success
What are metrics by which you would check for success?

## 8. Execution Plan
Describe the key tasks in executing your project, and in case of team project describe how will you partition the tasks.

## 9. Related Work
### 9.a. Papers
List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

### 9.b. Datasets
List datasets that you have identified and plan to use. Provide references (with full citation in the References section below).

### 9.c. Software
List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

## 10. References
List references correspondign to citations in your text above. For papers please include full citation and URL. For datasets and software include name and URL.

# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

This project introduces a driving assistance system aimed at improving road safety by monitoring and alerting drivers when violating the "3-second rule" distance between vehicles. Utilizing a Raspberry Pi 4, On-Board Diagnostics (OBD) for speed data, and a webcam, the system employs Python, YOLO for object detection, and a Kalman filter for error reduction. Results indicate a 94.1% accuracy in estimating distances, effectively warning drivers of proximity risks. This innovative integration of hardware and software presents a promising solution to enhance road safety and prevent rear-end collisions.

# 1. Introduction

* Motivation & Objective

  Ensuring a safe distance between vehicles while driving is critical to prevent collisions. In California, this guideline, known as the "3-second rule," dictates that the gap between your vehicle and the one in front should exceed the distance your current speed would cover in 3 seconds.

  Our objective is to create a driving assistance system that monitors the current speed of both your car and the one in front, while also gauging the distance separating the two vehicles. When this distance falls short of the 3-second rule, our system alerts the driver.

  Moreover, we aim to extend this concept to the vehicle behind as well. This entails detecting the distance and speed of the car trailing behind. If this distance is insufficient, a warning signal will be triggered to alert the trailing vehicle.

* State of the Art & Its Limitations

  Lidar technology currently dominates distance detection in vehicular safety due to its precision. However, its widespread adoption faces challenges. The high cost associated with lidar systems inhibits their broad use, particularly in budget-conscious applications. Additionally, lidar excels at distance estimation but lacks inherent object recognition capabilities. This limitation poses potential challenges, as the system may struggle to differentiate between various obstacles, impacting its overall effectiveness.

* Novelty & Rationale

  Our innovative approach replaces costly lidar technology with cost-effective cameras for front car identification and distance computation. This not only significantly reduces implementation expenses but also enhances versatility, enabling seamless integration with other camera-dependent features like Traffic Sign Recognition and lane keeping assistance. The ubiquity and ease of incorporating cameras align with modern vehicle trends, making our approach a practical and adaptable solution for enhancing vehicular safety.

* Potential Impact

  Cost-Efficiency:
  Our camera-based approach significantly reduces costs, broadening the reach of advanced driver-assistance systems to a wider audience.

  Ease of Implementation:
  Leveraging existing camera trends ensures a seamless and practical adoption of our solution in diverse vehicular environments.

  Technological Advancements:
  Our approach signifies a step forward in technological innovation, showcasing the practical use of existing hardware for enhanced safety systems.

* Challenges

  Resolution Constraints:
  The system grapples with limitations in camera resolution, impacting accurate object detection and distance calculation. Overcoming this challenge is imperative for achieving optimal system performance.
  
  Low-Light Conditions:
  In situations with inadequate lighting, the camera's capacity to capture clear and precise images may be compromised. Addressing challenges posed by low-light conditions requires effective solutions to ensure the system's reliability in various environmental settings.
  
  Recognition of Multiple Cars:
  The system encounters difficulties when confronted with multiple vehicles in close proximity. Accurately distinguishing and tracking multiple cars concurrently demands advanced algorithms and robust object recognition capabilities to prevent interference and ensure precise distance estimations.
  
  Issues with Lane Switching:
  Challenges arise when vehicles switch lanes abruptly or navigate complex traffic scenarios. The system must adeptly respond to sudden changes in the driving environment, ensuring consistent and accurate monitoring, even in dynamic situations involving lane changes.

* Requirements for Success

  Success in implementing the driving assistance system necessitates proficiency in computer vision, particularly using pretrained machine learning models like YOLO. Competence in setting up the Raspberry Pi 4, including hardware connections and software optimization, is vital. Seamless integration with On-Board Diagnostics (OBD), webcam plug-in simplicity, and adept Python coding skills complete the skill set required to achieve the project's goal of enhancing road safety through effective distance monitoring and driver alerts.

* Metrics of Success

  For distance estimation, critical metrics encompass the speed of processing, precision in identifying objects or individuals, and system reliability under diverse environmental conditions. As for the Kalman filter, the primary metric for success revolves around enhancing the accuracy of detected distances.

# 2. Related Work

# 3. Technical Approach

  The technical approach involves these key components for robust distance estimation:

* Distance Estimation:
  Utilizing YOLO for front car detection, the system determines the car's width within the frame. Leveraging the camera's focal length, the real width of the car and its frame width, the system calculates the distance between our car and the front vehicle. This method ensures precise distance estimations, even in dynamic scenarios.

* Kalman Filter:
  To enhance accuracy and mitigate errors arising from instances like YOLO's failure to detect, multiple cars, or excessively wide bounding boxes, a Kalman filter is employed. This filter dynamically adjusts distance estimations, providing a refined and more reliable output. This adaptive mechanism contributes to the system's overall robustness, ensuring consistent performance in varying conditions.

* Hardware Setup:
  The project is implemented on a Raspberry Pi 4 (RPI4), integrating an On-Board Diagnostics (OBD) adapter to acquire real-time car speed data and a USB webcam for visual input. This hardware ensemble forms the foundation for the successful execution of the proposed technical approach, combining the power of YOLO, the adaptability of the Kalman filter, and the comprehensive data inputs from the OBD and webcam components.

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References

[1] Welch, Greg, and Gary Bishop. "An introduction to the Kalman filter." (1995): 2. https://perso.crans.org/club-krobot/doc/kalman.pdf

[2] Zhang, Y., Wadhwa, N., Orts, S., Häne, C., Fanello, S., & Garg, R. (2020). Du2Net: Learning Depth Estimation from Dual-Cameras and Dual-Pixels. European Conference on Computer Vision.
https://arxiv.org/abs/2003.14299 

[3] Donghao Qiao, Farhana H. Zulkernine (2020). "Vision-based Vehicle Detection and Distance Estimation". Conference: 2020 IEEE Symposium Series on Computational Intelligence (SSCI). 

[4] https://github.com/Arun-purakkatt/medium_repo

[5] https://github.com/ultralytics/ultralytics.git

[6] https://github.com/eric612/Vehicle-Detection.git

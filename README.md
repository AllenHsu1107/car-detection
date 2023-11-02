# Front and Back Car Distance Detection

In this project, we aim to detect the distance between you and the car in front. If the distance is shorter than the required safe distance of your current speed, a warning is flashed to the drive.

Moreover, we can even further enhance this feature to the car behind. That is, if the distance between you and the car behind is shorter than safe distance of its current speed, we flash a light saying "TOO CLOSE" to the back car.

## Goal

To correctly detect the distance between two cars.
To correctly detect the speed of your car or the car behind.
To flash the warning if the required distance is not matched.

## Methods

### Detecting the distance between two cars

To detect the distance between two cars, there are 2 methods that come in handy:

1. Use two cameras shooting from differect direction, and calculate the distance by the difference between the angles.
2. Use one camera to catch the frame of the license plate, and calculate the distance by the size of the pixels.

For the back car case, since not all cars have license plates at the front end, method 1 is preferred.

### Detecting the speed of your car



### Detecting the speed of the car behind



### Flashing warning signs

This can be achieved by using basic Arduino circuits.


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def spring_mass(time_interval=0.1):
    def car1_motion(t):

        if t < car1_speed_up_time:
            displacement_car1 = 0.5 * car1_acceleration * (t**2)
        elif t >= car1_speed_up_time and t < car1_speed_up_time + car1_constant_time:
            displacement_car1 = car1_velocity * t -  0.5 * car1_acceleration * (car1_speed_up_time**2)
        else:
            displacement_car1 = car1_velocity * t -  0.5 * car1_acceleration * (car1_speed_up_time**2) - 0.5 * car1_acceleration * ((t - car1_constant_time - car1_speed_up_time)**2)
        return initial_distance + displacement_car1

    def car2_spring_system(y, t, m2, k, initial_distance):
        """
        Define the differential equation for the rear car (car 2) in the spring-mass system.
        y[0] is the displacement of car 2, y[1] is the velocity of car 2.
        """
        displacement_car1 = car1_motion(t)
        displacement_car2 = y[0]

        # Ensure that Car 2 cannot overtake Car 1
        # assert np.all(displacement_car2 <= displacement_car1), "Car 2 cannot overtake Car 1"

        dydt = [
            y[1],                                 # Velocity of car 2
            k * ((displacement_car1 - displacement_car2) - initial_distance) / m2  # Acceleration of car 2
        ]
        return dydt

    # System parameters
    m2 = 1000.0        # Mass of car 2
    k = 100.0        # Spring constant
    initial_distance = 40.0  # Initial distance between car 1 and car 2
    # Initial conditions: displacement and velocity for car 2
    initial_conditions_car2 = [0.0, 0.0]
    # Time grid
    car1_acceleration = 2.5
    car1_velocity = 11
    car1_constant_time = 20
    car1_speed_up_time = car1_velocity / car1_acceleration
    time = car1_constant_time + car1_speed_up_time * 2
    num_data_point = int(time / time_interval) + 1
    t = np.linspace(0, time, num_data_point)

    # Solve the differential equation for car 2
    solution_car2 = odeint(car2_spring_system, initial_conditions_car2, t, args=(m2, k, initial_distance))

    # Extract displacement and velocity of car 2 from the solution
    displacement_car2 = solution_car2[:, 0]
    velocity_car2 = solution_car2[:, 1]
    displacement_car1 = []
    for single_t in t:
        displacement_car1.append(car1_motion(single_t))
    # Plot the positions of car 1 (independent motion) and car 2 (spring-mass system) over time
    # plt.figure(figsize=(8, 6))
    # plt.plot(t, displacement_car1, label='Car 1 (front car)')
    # plt.plot(t, displacement_car2, label='Car 2 (rear car)')
    # plt.title('Two-Car System: Car 1 (Independent) and Car 2 (Spring-Mass)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Displacement (m)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    true_distance = displacement_car1 - displacement_car2
    time_intervals = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    return true_distance, time_intervals

if __name__ == "__main__":
    spring_mass()

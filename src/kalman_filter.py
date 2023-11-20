import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def kalman_filter(observed_distance, time_intervals, process_variance, measurement_variance):
    initial_distance = observed_distance[0]
    initial_velocity = 0.0
    x = np.array([[initial_distance], [initial_velocity]])  # State vector [distance, velocity]
    P = np.diag([1, 1])  # Initial covariance matrix
    H = np.array([[1, 0]])  # Measurement matrix
    R = np.array([[measurement_variance]])  # Measurement covariance
    estimated_distance = []
    for i in range(len(observed_distance)):
        # Prediction step
        delta_t = time_intervals[i] if i > 0 else 1.0  # Use 1.0 as a default for the first iteration
        F = np.array([[1, delta_t], [0, 1]])  # State transition matrix
        Q = np.array([[0.5 * delta_t**2 * process_variance, delta_t * process_variance],
                      [delta_t * process_variance, process_variance]])  # Process covariance
        x_hat = np.dot(F, x)
        P_hat = np.dot(np.dot(F, P), F.T) + Q
        # Update step
        y = observed_distance[i] - np.dot(H, x_hat)
        S = np.dot(np.dot(H, P_hat), H.T) + R
        K = np.dot(np.dot(P_hat, H.T), np.linalg.inv(S))
        x = x_hat + np.dot(K, y)
        P = np.dot((np.eye(2) - np.dot(K, H)), P_hat)
        # Save the estimated distance
        estimated_distance.append(x[0, 0])

    return estimated_distance

np.random.seed(42)
true_distance = np.linspace(0, 100, 100)
observed_distance = true_distance + np.random.normal(0, 5, 100)
time_intervals = np.random.uniform(0.5, 1.5, 100)

def objective_function(params):
    process_variance, measurement_variance = params
    estimated_distance = kalman_filter(observed_distance, time_intervals, process_variance, measurement_variance)
    mse = np.mean((true_distance - estimated_distance)**2)
    return mse

initial_guess = [0.1, 5.0]
bounds = [(1e-5, 1.0), (1e-5, 10.0)]
result = minimize(objective_function, initial_guess, bounds=bounds)
best_process_variance, best_measurement_variance = result.x
estimated_distance_optimized = kalman_filter(observed_distance, time_intervals, best_process_variance, best_measurement_variance)
estimated_distance_unoptimized = kalman_filter(observed_distance, time_intervals, initial_guess[0], initial_guess[1])
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(true_distance, label='True Distance')
plt.plot(observed_distance, label='Observed Distance', marker='o')
plt.plot(estimated_distance_optimized, label='Optimized Estimated Distance', linestyle='--', marker='x')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Optimized Kalman Filter for Distance Estimation')
plt.subplot(1, 2, 2)
plt.plot(true_distance, label='True Distance')
plt.plot(observed_distance, label='Observed Distance', marker='o')
plt.plot(estimated_distance_unoptimized, label='Unoptimized Estimated Distance', linestyle='--', marker='x')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Unoptimized Kalman Filter for Distance Estimation')
plt.tight_layout()
plt.show()
print("Best Process Variance (Optimized):", best_process_variance)
print("Best Measurement Variance (Optimized):", best_measurement_variance)
mse_observed = np.mean((true_distance - observed_distance)**2)
mse_optimized = np.mean((true_distance - estimated_distance_optimized)**2)
mse_unoptimized = np.mean((true_distance - estimated_distance_unoptimized)**2)
print("\nMean Squared Error (Observed Distance):", mse_observed)
print("Mean Squared Error (Optimized Estimated Distance):", mse_optimized)
print("Mean Squared Error (Unoptimized Estimated Distance):", mse_unoptimized)

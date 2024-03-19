import numpy as np
from scipy import io
from ukf import *
from quaternion2 import Quaternion
from load_data import *


def estimate_rot(data_num=1):
   # Load data
    imu_timestamps, acceleration, gyroscope = load_imu_data(data_num)

    last_quat = Quaternion()
    last_covariance = np.identity(3) * 0.1  # Last covariance in vector form
    process_noise_cov = np.identity(3) * 5  # Process noise covariance
    measurement_noise_cov = np.identity(3) * 5  # Measurement noise covariance

    num_timestamps = imu_timestamps.shape[0]
    estimated_euler_angles = np.zeros((num_timestamps, 3))


    for t in range(num_timestamps):
        # Prediction
        sigma_points = compute_sigma_pts(last_quat, last_covariance, process_noise_cov)

        processed_sigma_points = process_model(sigma_points, gyroscope[t])

        quat_predicted, pred_covariance, sigma_weights = prediction(processed_sigma_points, last_quat)

        measurement_residual, measurement_residual_cov, cross_covariance = measurement_model(processed_sigma_points, acceleration[t], sigma_weights, measurement_noise_cov)

        # Update
        kalman_gain = np.dot(cross_covariance, np.linalg.inv(measurement_residual_cov))
        last_quat, last_covariance = update(quat_predicted, pred_covariance, measurement_residual, measurement_residual_cov, kalman_gain)
        
        estimated_euler_angles[t] = last_quat.normalize().euler_angles()
    
    return estimated_euler_angles[:,0], estimated_euler_angles[:,1], estimated_euler_angles[:,2]
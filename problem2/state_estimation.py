import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from ukf import *
from quaternion2 import Quaternion
from load_data import *

def estimate_rotation(data_num=1):
    # Load data
    imu_timestamps, acceleration, gyroscope = load_imu_data(data_num)
    vicon_timestamps, vicon_euler_angles = load_vicon_data(data_num)

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
        
    orientation_plot(imu_timestamps, estimated_euler_angles, vicon_timestamps, vicon_euler_angles)

def orientation_plot(imu_timestamps, ukf_estimated_euler, vicon_timestamps, vicon_euler):
    plt.figure(3)
    plt.subplot(3, 1, 1)
    plt.plot(vicon_timestamps, vicon_euler[:, 2], label='Vicon')
    plt.plot(imu_timestamps, ukf_estimated_euler[:, 2], label='UKF')
    plt.title('Roll')
    plt.ylabel('Angle')


    plt.subplot(3, 1, 2)
    plt.plot(vicon_timestamps, vicon_euler[:, 1], label='Vicon')
    plt.plot(imu_timestamps, ukf_estimated_euler[:, 1], label='UKF')
    plt.title('Pitch')
    plt.ylabel('Angle')
    
    plt.subplot(3, 1, 3)
    plt.plot(vicon_timestamps,vicon_euler[:, 0], label='Vicon')
    plt.plot(imu_timestamps,ukf_estimated_euler[:, 0], label='UKF')
    plt.title('Yaw')
    plt.ylabel('Angle')
    
    plt.tight_layout()


    plt.subplots_adjust(hspace=0.5) 

    plt.show()

if __name__ == '__main__':
    estimate_rotation(1)

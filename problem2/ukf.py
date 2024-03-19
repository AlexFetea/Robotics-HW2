import numpy as np
from quaternion2 import Quaternion

def compute_sigma_pts(quaternion_state, state_covariance, process_noise_covariance):
    """Computes sigma points for the Unscented Kalman Filter."""
    state_dimension = state_covariance.shape[0]  # Dimension of the state space
    cholesky_factor = np.linalg.cholesky(state_covariance + process_noise_covariance) * np.sqrt(2 * state_dimension)  # Calculation of S
    
    sigma_point_offsets = np.hstack((cholesky_factor, -cholesky_factor))  # Concatenation of S and -S

    
    # print(sigma_point_offsets)
    sigma_pts = np.empty((2 * state_dimension,), dtype=object)
    for i in range(2 * state_dimension):
        sigma_quat = Quaternion()
        sigma_quat.from_axis_angle(sigma_point_offsets[:, i])
        
        sigma_pts[i] = quaternion_state * sigma_quat  # Quaternion rotation applied

    sigma_pts_with_mean = np.insert(sigma_pts, 0, quaternion_state)  # Insert mean as first sigma point

    return sigma_pts_with_mean

def process_model(sigma_pts, gyroscope_measurement, delta_time):
    """Processes sigma points through the dynamic model."""
    n = sigma_pts.shape[0]
    processed_sigma_pts = np.empty(n, dtype=object)
    
    gyroscope_delta_quat = Quaternion()
    gyroscope_delta_quat.from_axis_angle(gyroscope_measurement * delta_time)  # Compute once for efficiency

    for i in range(n):
        processed_sigma_pts[i] = sigma_pts[i] * gyroscope_delta_quat  # Apply gyro rotation

    return processed_sigma_pts

def prediction(processed_sigma_pts, quat_prior):
    """Predicts the next state."""
    n = len(processed_sigma_pts)
    quat_predicted, sigma_weights = quat_avg(processed_sigma_pts, quat_prior)  # Quaternion average calculation

    predicted_covariance = np.zeros((3, 3))  # Initialize prediction covariance matrix
    for i in range(n):
        predicted_covariance += np.outer(sigma_weights[i, :], sigma_weights[i, :])
    predicted_covariance /= n  # Normalize by number of points

    return quat_predicted, predicted_covariance, sigma_weights

def measurement_model(processed_sigma_pts, acceleration_measurement, sigma_weights, measurement_noise_covariance):
    """Models the measurement prediction."""
    n = processed_sigma_pts.shape[0]
    world_gravity_quat = Quaternion(scalar=0, vec=[0, 0, 1])  # World gravity as quaternion

    predicted_accelerations = np.zeros((n, 3))
    for i in range(n):
        # Rotation of gravity to body frame for each sigma point
        predicted_accelerations[i] = np.array((processed_sigma_pts[i].inv() * world_gravity_quat * processed_sigma_pts[i]).vec())

    mean_acceleration = np.mean(predicted_accelerations, axis=0)
    mean_acceleration /= np.linalg.norm(mean_acceleration)  # Normalization

    # Covariance matrices initialization
    measurement_covariance = np.zeros((3, 3))
    cross_covariance = np.zeros((3, 3))
    acc_error = predicted_accelerations - mean_acceleration
    for i in range(n):
        measurement_covariance += np.outer(acc_error[i, :], acc_error[i, :])
        cross_covariance += np.outer(sigma_weights[i, :], acc_error[i, :])
    measurement_covariance /= n
    cross_covariance /= n

    acceleration_measurement /= np.linalg.norm(acceleration_measurement)  # Measured acceleration normalization
    measurement_innovation = acceleration_measurement - mean_acceleration
    innovation_covariance = measurement_covariance + measurement_noise_covariance  # Measurement noise covariance addition

    return measurement_innovation, innovation_covariance, cross_covariance

def update(quat_predicted, predicted_covariance, measurement_innovation, innovation_covariance, kalman_gain):
    """Updates the state with the measurement."""
    quat_gain = Quaternion()
    quat_gain.from_axis_angle(kalman_gain.dot(measurement_innovation))  # Convert to quaternion
    quat_updated = quat_gain * quat_predicted  # State update
    
    updated_covariance = predicted_covariance - kalman_gain.dot(innovation_covariance).dot(kalman_gain.T)  # Covariance update

    return quat_updated, updated_covariance

def quat_avg(quaternions, initial_estimate):
    """Computes the average of a set of quaternions."""
    n = len(quaternions)
    epsilon = 1E-3  # Convergence criterion
    max_iter = 100  # Maximum number of iterations
    average_quat = initial_estimate

    for _ in range(max_iter):
        error_vectors = np.zeros((n, 3))
        for i in range(n):
            quat_error = quaternions[i] * average_quat.inv()  # Error quaternion
            quat_error.normalize()
            error_vector = quat_error.axis_angle()

            error_norm = np.linalg.norm(error_vector)
            if error_norm == 0:
                error_vectors[i, :] = np.zeros(3)
            else:
                error_vectors[i, :] = (-np.pi + np.mod(error_norm + np.pi, 2 * np.pi)) / error_norm * error_vector

        mean_error = np.mean(error_vectors, axis=0)  # Update estimate
        average_quat = Quaternion().from_axis_angle(mean_error) * average_quat
        average_quat.normalize()

        if np.linalg.norm(mean_error) < epsilon:  # Check convergence
            break

    return average_quat, error_vectors

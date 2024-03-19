import numpy as np
from scipy import io
from quaternion2 import Quaternion


def load_imu_data(index):
    # Load IMU data
    imu_data = io.loadmat("imu/imuRaw" + str(index) + ".mat")
    accel_raw = np.array(imu_data['vals'])[0:3].astype('float64')
    gyro_raw = np.array(imu_data['vals'])[3:6]
    
    imu_timestamps = np.array(imu_data['ts']).T
    

    acc_bias = [510.9, 501, 505.95]
    accel_sensitivity = 32.61319917135248
    acceleration_calibrated = (accel_raw.T - acc_bias) * 3300 / 1023 / accel_sensitivity
    
    acceleration_calibrated[:,0] *= -1
    acceleration_calibrated[:,1] *= -1
    
    

    gyro = np.array([gyro_raw[1,:], gyro_raw[2,:], gyro_raw[0,:]])
    gyro_bias = np.array([373.6, 375.2, 369.8])
    gyro_sensitivity = 321.42265210121775    
    gyroscope_calibrated = (gyro.T - gyro_bias) * 3300 / 1023 * ((np.pi / 180) / gyro_sensitivity)
    
    return imu_timestamps, acceleration_calibrated, gyroscope_calibrated

def load_vicon_data(index):
    vicon_data = io.loadmat("vicon/viconRot" + str(index) + ".mat")
    vicon_rotations = np.array(vicon_data['rots'])
    vicon_timestamps = np.array(vicon_data['ts']).T

    num_rotations = np.shape(vicon_rotations)[2]
    vicon_euler_angles = np.zeros((num_rotations,3))
    for i in range(num_rotations):
        rotation_matrix = vicon_rotations[:,:,i]
        q = Quaternion()
        q.from_rotm(rotation_matrix)
        vicon_euler_angles[i] = q.euler_angles()

    return vicon_timestamps, vicon_euler_angles

if __name__ == "__main__":
    load_imu_data(1)

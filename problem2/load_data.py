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
    # print((accel_raw.T - acc_bias))
    acceleration_calibrated = (accel_raw.T - acc_bias) * 3300 / 1023 / accel_sensitivity
    
    acceleration_calibrated[:,0] *= -1
    acceleration_calibrated[:,1] *= -1
    
    # import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(20, 4))
    # plt.plot(acceleration_calibrated.T[0], label='x', linestyle='--')
    # plt.plot(acceleration_calibrated.T[1], label='y', linestyle='--')
    # plt.plot(acceleration_calibrated.T[2], label='z', linestyle='--')
    # plt.title('Accelerometer Z')
    # plt.xlabel('Sample')
    # plt.ylabel('Acceleration (m/s^2)')
    # plt.legend()
    # plt.show()

    

    gyro = np.array([gyro_raw[1,:], gyro_raw[2,:], gyro_raw[0,:]])

    gyro_sensitivity = 3.30
    gyro_bias = np.array([373.6, 375.2, 369.8])
    print(((np.pi / 180) / gyro_sensitivity))
    gyroscope_calibrated = (gyro.T - gyro_bias) * 3300 / 1023 * ((np.pi / 180) / gyro_sensitivity)
    
    gyro_pos=[]
    cur = Quaternion()  
    for i in range(gyroscope_calibrated.T.shape[1]):
        q = Quaternion().from_axis_angle(gyroscope_calibrated[i])
        cur *= q
        gyro_pos.append(cur.euler_angles())
        
    gyro_pos = np.array(gyro_pos)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 4))
    plt.plot(gyro_pos.T[0], label='x', linestyle='--')
    plt.plot(gyro_pos.T[1], label='y', linestyle='--')
    plt.plot(gyro_pos.T[2], label='z', linestyle='--')
    plt.title('Accelerometer Z')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.show()

    
    imu_measurements = np.hstack((acceleration_calibrated, gyroscope_calibrated))

    return imu_timestamps, imu_measurements

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

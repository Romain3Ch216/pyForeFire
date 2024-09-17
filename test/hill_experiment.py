import numpy as np
import math
import tensorflow as tf
from forefire_TF_helpers import save_model_structure2


def random_fire_front(domain_width, domain_height, size=0.05):
    x_axis = np.linspace(0, domain_width, 1000)
    y_axis = np.linspace(0, domain_width, 1000)
    x = min(
        max(2 * size * domain_width, x_axis[np.random.randint(0, len(x_axis))]),
        domain_width - 2 * size * domain_width)
    y = min(
        max(2 * size * domain_height, y_axis[np.random.randint(0, len(y_axis))]),
        domain_height - 2 * size * domain_height)
    

    fire_front = [
        [x - size * domain_width, y + size * domain_height],
        [x, y],
        [x - size * domain_width, y - size * domain_height]
    ]
    
    return fire_front

def init_emulator(inputs, emulator_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='relu', input_shape=(1,))
    ])
    save_model_structure2(model, emulator_path, inputs, 'ROS')

def random_wind_field(wind_speed=[0, 30], angle=[0, 360]):
    """
    Args:
        - wind_speed (list): range of wind speed in m/s
        - angle (list): range of wind direction in degrees
    """
    wind_speed = np.linspace(wind_speed[0], wind_speed[1], 1000)
    angle = np.linspace(angle[0], angle[1], 1000)
    wind_speed = wind_speed[np.random.randint(0, len(wind_speed))]
    angle = angle[np.random.randint(0, len(angle))]

    rotation_angle_rad = math.radians(angle)
    windU = wind_speed * math.cos(rotation_angle_rad)
    windY = wind_speed * math.sin(rotation_angle_rad)
    return windU, windY

def hill(x, mean, cov, height=1000):
    N = len(mean)
    den = (2*np.pi)**(N/2) * np.linalg.det(cov)**0.5
    exp = np.exp(-0.5 * np.einsum('...k,kl,...l->...', x-mean, np.linalg.inv(cov), x-mean))
    gaussian = exp / den
    altitude = height * (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    return altitude

def isotropic_hill(domain_width, domain_height, height=1000):
    mean = np.array([domain_height // 2, domain_width //2])

    cov = np.array([
        [1e5, 0],
        [0, 1e5]])

    map_x, map_y = np.meshgrid(
        np.arange(domain_height),
        np.arange(domain_width)
    )

    map = np.empty(map_x.shape + (2,))
    map[:, :, 0] = map_x
    map[:, :, 1] = map_y

    altitude_map = hill(map, mean, cov, height)
    return altitude_map 
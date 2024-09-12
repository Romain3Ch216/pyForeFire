import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime
import tensorflow as tf
from forefire_TF_helpers import save_model_structure2
from simulation import UniformWindForeFireSimulation
from forefire_helper import get_fuels_table
import xarray as xr
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('BuildDB')


def main(
    db_folder,
    propagation_model,
    model_inputs,
    domain_width,
    domain_height,
    fuel_type,
    nb_steps,
    step_size,
    run_id
    ):
    logger.info(f'Run simulation {run_id} to build fake observations.')
    emulator_path, simulation_inputs = \
        run_simulation(
            db_folder,
            propagation_model,
            model_inputs,
            domain_width,
            domain_height,
            fuel_type,
            nb_steps,
            step_size,
            run_id
        )
    horizontal_wind, vertical_wind, altitude_map, fuel_map, fire_front = simulation_inputs

    logger.info(f'Build training data set from simulation {run_id}.')
    make_db(
        db_folder,
        emulator_path,
        horizontal_wind,
        vertical_wind,
        altitude_map,
        fuel_map,
        fire_front,
        nb_steps,
        step_size,
        run_id)


def run_simulation(
    db_folder,
    propagation_model,
    model_inputs,
    domain_width,
    domain_height,
    fuel_type,
    nb_steps,
    step_size,
    run_id
    ):
    emulator_path = os.path.join(db_folder, propagation_model + '.ffann')
    if not os.path.exists(emulator_path):
        init_emulator(model_inputs, emulator_path)

    fuels_table = get_fuels_table(propagation_model)

    horizontal_wind, vertical_wind = random_wind_field()
    altitude_map = isotropic_hill(domain_width, domain_height)
    fuel_map = fuel_type * np.ones_like(altitude_map)
    fire_front = random_fire_front(domain_width, domain_height)

    simulation = UniformWindForeFireSimulation(
        propagation_model,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_map,
        altitude_map,
        fire_front
    )

    simulation(nb_steps, step_size)

    simulation.ff['caseDirectory'] = '/'.join(db_folder.split('/')[:-1])
    simulation.ff['fireOutputDirectory'] = db_folder.split('/')[-1]
    simulation.ff['experiment'] = f'simulation_{run_id}'

    simulation.ff.execute("save[]")

    return emulator_path, (horizontal_wind, vertical_wind, altitude_map, fuel_map, fire_front)

def make_db(
        db_folder,
        emulator_path,
        horizontal_wind,
        vertical_wind,
        altitude_map,
        fuel_map,
        fire_front,
        nb_steps,
        step_size,
        run_id
    ):
    logger_path = os.path.join(db_folder, f'simulation_{run_id}_db.csv')
    fire_observation = os.path.join(db_folder, f'simulation_{run_id}.0.nc')
    propagation_model = "BMapLoggerForANNTraining"
    fuels_table = get_fuels_table(emulator_path.split('/')[-1].split('.')[0])

    simulation = UniformWindForeFireSimulation(
        propagation_model,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_map,
        altitude_map,
        fire_front,
        fire_observation,
        logger_path,
        emulator_path
    )

    simulation(nb_steps, step_size)



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

def hill(x, mean, cov, height=100):
    N = len(mean)
    den = (2*np.pi)**(N/2) * np.linalg.det(cov)**0.5
    exp = np.exp(-0.5 * np.einsum('...k,kl,...l->...', x-mean, np.linalg.inv(cov), x-mean))
    gaussian = exp / den
    altitude = height * (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    return altitude

def isotropic_hill(domain_width, domain_height, height=100):
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


if __name__ == '__main__':
    db_folder = '/home/ai4geo/Documents/nn_ros_models/hill_experiments'
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    propagation_model = 'RothermelAndrews2018'
    model_inputs = [
        "fuel.fl1h_tac",
        "fuel.fd_ft",
        "fuel.Dme_pc",
        "fuel.SAVcar_ftinv",
        "fuel.H_BTUlb",
        "fuel.totMineral_r",
        "fuel.effectMineral_r",
        "fuel.fuelDens_lbft3",
        "fuel.mdOnDry1h_r",
        "normalWind",
        "slope"
        ]
    domain_width = 1000
    domain_height = 1000
    fuel_type = 6
    n_simulations = 5
    nb_steps = 5
    step_size = 10
    run_id = sys.argv[1]

    main(
        db_folder,
        propagation_model,
        model_inputs,
        domain_width,
        domain_height,
        fuel_type,
        nb_steps,
        step_size,
        run_id
    )
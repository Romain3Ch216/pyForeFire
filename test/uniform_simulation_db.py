import numpy as np
import os
from simulation import UniformForeFireSimulation
from forefire_helper import get_fuels_table
import sys
import logging
from hill_experiment import *
import json

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
        db_folder,
        emulator_path,
        domain,
        horizontal_wind,
        vertical_wind,
        slope,
        fire_front,
        nb_steps,
        step_size,
        perimeter_resolution,
        minimal_propagative_front_depth,
        spatial_increment,
        look_ahead_distance_for_time_gradient,
        run_id
    ):
    logger.info(f'Build training data set from simulation {run_id}.')
    logger_path = os.path.join(db_folder, f'simulation_{run_id}_db.csv')
    fire_observation = os.path.join(db_folder, f'simulation_{run_id}.0.nc')
    propagation_model = "BMapLoggerForANNTraining"
    fuels_table = get_fuels_table(emulator_path.split('/')[-1].split('.')[0])

    simulation = UniformForeFireSimulation(
        propagation_model,
        domain,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_type,
        slope,
        fire_front,
        fire_observation,
        logger_path,
        emulator_path,
        perimeter_resolution=perimeter_resolution,
        minimal_propagative_front_depth=minimal_propagative_front_depth,
        spatial_increment=spatial_increment,
        look_ahead_distance_for_time_gradient=look_ahead_distance_for_time_gradient
    )

    pathes = simulation(nb_steps, step_size)

    # from forefire_helper import plot_simulation
    # plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], myExtents=None)


if __name__ == '__main__':
    db_folder = '/home/ai4geo/Documents/nn_ros_models/uniform_experiments'
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    propagation_model = 'RothermelAndrews2018'
    model_inputs = [
        # "fuel.fl1h_tac",
        # "fuel.fd_ft",
        # "fuel.Dme_pc",
        # "fuel.SAVcar_ftinv",
        # "fuel.H_BTUlb",
        # "fuel.totMineral_r",
        # "fuel.effectMineral_r",
        # "fuel.fuelDens_lbft3",
        # "fuel.mdOnDry1h_r",
        "normalWind",
        "slope",
        "nodeLocationX",
        "nodeLocationY",
        "nodeID",
        "nodeTime"
        ]
    
    domain_width = 1000
    domain_height = 1000
    domain = (0, 0, domain_width, domain_height)
    horizontal_wind, vertical_wind = 0.0, 0
    fuel_type = 6

    run_id = sys.argv[1]

    nb_steps = 10
    step_size = 20
    perimeter_resolution = 1
    spatial_increment = 1
    minimal_propagative_front_depth = 1
    look_ahead_distance_for_time_gradient = 2

    with open(os.path.join(db_folder, f'config_{run_id}.yaml'), 'r') as stream:
        config = json.load(stream)

    main(
        db_folder,
        config['emulator_path'],
        domain,
        horizontal_wind,
        vertical_wind,
        config['slope'],
        config['fire_front'],
        nb_steps,
        step_size,
        perimeter_resolution,
        minimal_propagative_front_depth,
        spatial_increment,
        look_ahead_distance_for_time_gradient,
        run_id
    )
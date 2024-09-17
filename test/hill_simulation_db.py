import os
import sys
import logging
import json
import numpy as np

from simulation import UniformWindForeFireSimulation
from forefire_helper import plot_simulation
from forefire_helper import get_fuels_table
from hill_experiment import *


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    db_folder,
    emulator_path,
    domain_width,
    domain_height,
    horizontal_wind,
    vertical_wind,
    fuel_type,
    fire_front,
    run_id,
    plot=False
    ):
    """
    Run a forced HILL experiment given domain dimensions, fuel_map, wind and initial fire front.
    """
    # Fixed metadata
    logger_path = os.path.join(db_folder, f'simulation_{run_id}_db.csv')
    fire_observation = os.path.join(db_folder, f'simulation_{run_id}.0.nc')
    
    # Fixed parameters
    spatial_increment = 1
    perimeter_resolution = 5
    minimal_propagative_front_depth = 10
    look_ahead_distance_for_time_gradient = 20
    relax = 0.5

    propagation_model = "BMapLoggerForANNTraining"

    fuels_table = get_fuels_table(emulator_path.split('/')[-1].split('.')[0])
    altitude_map = isotropic_hill(domain_width, domain_height)
    fuel_map = fuel_type * np.ones_like(altitude_map)

    nb_steps = 10
    step_size = 20

    ##---Simulation---##
    logger.info(f'Run training dataset from forced simulation {run_id}.')

    simulation = UniformWindForeFireSimulation(
        propagation_model,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_map,
        altitude_map,
        fire_front,
        fire_observation=fire_observation,
        logger_path=logger_path,
        model_path=emulator_path,
        spatial_increment=spatial_increment,
        perimeter_resolution=perimeter_resolution,
        minimal_propagative_front_depth=minimal_propagative_front_depth,
        look_ahead_distance_for_time_gradient=look_ahead_distance_for_time_gradient,
        relax=relax
    )

    pathes = simulation(nb_steps, step_size)

    if plot:
        plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], myExtents=None)
    


if __name__ == '__main__':
    run_id = sys.argv[1]
    db_folder = '/home/ai4geo/Documents/nn_ros_models/hill_experiments'
    
    # Random parameters
    with open(os.path.join(db_folder, f'config_{run_id}.json'), 'r') as stream:
        config = json.load(stream)
    
    main(
        db_folder,
        config['emulator_path'],
        config['domain_width'],
        config['domain_height'],
        config['horizontal_wind'],
        config['vertical_wind'],
        config['fuel_type'],
        config['fire_front'],
        run_id
    )

# import numpy as np
# import os
# from simulation import UniformWindForeFireSimulation
# from forefire_helper import get_fuels_table
# import sys
# import logging
# from hill_experiment import *
# import json

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger(__name__)

# def main(
#         db_folder,
#         emulator_path,
#         horizontal_wind,
#         vertical_wind,
#         fire_front,
#         nb_steps,
#         step_size,
#         run_id
#     ):
#     logger.info(f'Build training data set from simulation {run_id}.')

#     altitude_map = isotropic_hill(domain_width, domain_height)
#     fuel_map = fuel_type * np.ones_like(altitude_map)

#     simulation = UniformWindForeFireSimulation(
#         propagation_model,
#         fuels_table,
#         horizontal_wind,
#         vertical_wind,
#         fuel_map,
#         altitude_map,
#         fire_front,
#         fire_observation,
#         logger_path,
#         emulator_path
#     )

#     pathes = simulation(nb_steps, step_size)

#     # from forefire_helper import plot_simulation
#     # plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], myExtents=None)


# if __name__ == '__main__':
#     db_folder = '/home/ai4geo/Documents/nn_ros_models/hill_experiments'
#     if not os.path.exists(db_folder):
#         os.makedirs(db_folder)
#     propagation_model = 'RothermelAndrews2018'
#     model_inputs = [
#         "fuel.fl1h_tac",
#         "fuel.fd_ft",
#         "fuel.Dme_pc",
#         "fuel.SAVcar_ftinv",
#         "fuel.H_BTUlb",
#         "fuel.totMineral_r",
#         "fuel.effectMineral_r",
#         "fuel.fuelDens_lbft3",
#         "fuel.mdOnDry1h_r",
#         "normalWind",
#         "slope"
#         ]
#     domain_width = 1000
#     domain_height = 1000
#     fuel_type = 6
#     nb_steps = 20
#     step_size = 10
#     run_id = sys.argv[1]
    
#     with open(os.path.join(db_folder, f'config_{run_id}.yaml'), 'r') as stream:
#         config = json.load(stream)

#     main(
#         db_folder,
#         config['emulator_path'],
#         config['horizontal_wind'],
#         config['vertical_wind'],
#         config['fire_front'],
#         nb_steps,
#         step_size,
#         run_id
#     )
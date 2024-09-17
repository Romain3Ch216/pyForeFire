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
    domain_width,
    domain_height,
    fuel_map,
    wind,
    fire_front,
    run_id,
    plot=False
    ):
    """
    Run a HILL experiment given domain dimensions, fuel_map, wind and initial fire front.
    """
    # Fixed metadata
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
        "slope",
        "nodeLocationX",
        "nodeLocationY",
        "nodeID",
        "nodeTime"
        ]
    
    # Fixed parameters
    spatial_increment = 1
    perimeter_resolution = 5
    minimal_propagative_front_depth = 10
    look_ahead_distance_for_time_gradient = 20
    relax = 0.5

    fuels_table = get_fuels_table(propagation_model)
    altitude_map = isotropic_hill(domain_width, domain_height)

    nb_steps = 10
    step_size = 20

    # Random parameters
    horizontal_wind, vertical_wind = wind

    # Initialize emulator with model_inputs
    emulator_path = os.path.join(db_folder, propagation_model + '.ffann')
    if not os.path.exists(emulator_path):
        init_emulator(model_inputs, emulator_path)

    ##---Simulation---##
    logger.info(f'Run simulation {run_id} to build fake observations.')

    simulation = UniformWindForeFireSimulation(
        propagation_model,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_map,
        altitude_map,
        fire_front,
        spatial_increment=spatial_increment,
        perimeter_resolution=perimeter_resolution,
        minimal_propagative_front_depth=minimal_propagative_front_depth,
        look_ahead_distance_for_time_gradient=look_ahead_distance_for_time_gradient,
        relax=relax
    )

    pathes = simulation(nb_steps, step_size)

    if plot:
        plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], myExtents=None)
    
    simulation.ff['caseDirectory'] = '/'.join(db_folder.split('/')[:-1])
    simulation.ff['fireOutputDirectory'] = db_folder.split('/')[-1]
    simulation.ff['experiment'] = f'simulation_{run_id}'

    simulation.ff.execute("save[]")

    config = {
        'emulator_path': emulator_path,
        'domain_width': domain_width,
        'domain_height': domain_height,
        'horizontal_wind': horizontal_wind,
        'vertical_wind': vertical_wind,
        'fuel_type': fuel_type,
        'fire_front': fire_front
    }

    with open(os.path.join(db_folder, f'config_{run_id}.json'), 'w') as outfile:
        json.dump(config, outfile)


if __name__ == '__main__':
    run_id = sys.argv[1]
    db_folder = '/home/ai4geo/Documents/nn_ros_models/hill_experiments'
    
    domain_width = 2000
    domain_height = 2000
    
    # Random parameters
    wind = random_wind_field()
    fuel_type = np.random.randint(1, 10)
    fuel_map = fuel_type * np.ones((domain_height, domain_width))
    fire_front = random_fire_front(domain_width, domain_height)
    
    main(
        db_folder,
        domain_width,
        domain_height,
        fuel_map,
        wind,
        fire_front,
        run_id,
        # plot=True
    )
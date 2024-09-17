import numpy as np
import os

from simulation import UniformForeFireSimulation
from forefire_helper import get_fuels_table
import sys
import logging
import json
from hill_experiment import *


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    db_folder,
    propagation_model,
    model_inputs,
    domain,
    horizontal_wind,
    vertical_wind,
    slope,
    fuel_type,
    nb_steps,
    step_size,
    perimeter_resolution,
    minimal_propagative_front_depth,
    spatial_increment,
    look_ahead_distance_for_time_gradient,
    run_id
    ):
    logger.info(f'Run simulation {run_id} to build fake observations.')
    emulator_path = os.path.join(db_folder, propagation_model + '.ffann')
    if not os.path.exists(emulator_path):
        init_emulator(model_inputs, emulator_path)

    fuels_table = get_fuels_table(propagation_model)
    # fire_front = random_fire_front(domain[2], domain[3])
    fire_front = [[450, 550], [500, 500], [450, 450]]

    simulation = UniformForeFireSimulation(
        propagation_model,
        domain,
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_type,
        slope,
        fire_front,
        perimeter_resolution=perimeter_resolution,
        minimal_propagative_front_depth=minimal_propagative_front_depth,
        spatial_increment=spatial_increment,
        look_ahead_distance_for_time_gradient=look_ahead_distance_for_time_gradient
    )

    pathes = simulation(nb_steps, step_size)

    plotExtents = (
        float(simulation.ff["SWx"]),
        float(simulation.ff["SWx"]) + float(simulation.ff["Lx"]),
        float(simulation.ff["SWy"]),
        float(simulation.ff["SWy"]) + float(simulation.ff["Ly"]))
    
    # from forefire_helper import plot_simulation
    # plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], plotExtents, None, title=propagation_model) #, save_exp=os.path.join(db_folder, f'sim_{run_id}.pdf'))
    
    simulation.ff['caseDirectory'] = '/'.join(db_folder.split('/')[:-1])
    simulation.ff['fireOutputDirectory'] = db_folder.split('/')[-1]
    simulation.ff['experiment'] = f'simulation_{run_id}'

    simulation.ff.execute("save[]")

    config = {
        'emulator_path': emulator_path,
        'slope': slope,
        'fire_front': [[x[0], x[1]] for x in fire_front]
    }

    with open(os.path.join(db_folder, f'config_{run_id}.yaml'), 'w') as outfile:
        json.dump(config, outfile)


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
    run_id = sys.argv[1]
    domain_width = 1000
    domain_height = 1000
    domain = (0, 0, domain_width, domain_height)
    horizontal_wind, vertical_wind = 0.0, 0.0
    slopes = np.linspace(-40, 40, 20)
    slope = slopes[int(run_id)-1]
    fuel_type = 6

    nb_steps = 10
    step_size = 20
    perimeter_resolution = 1
    spatial_increment = 1
    minimal_propagative_front_depth = 1
    look_ahead_distance_for_time_gradient = 2

    main(
        db_folder,
        propagation_model,
        model_inputs,
        domain,
        horizontal_wind,
        vertical_wind,
        slope,
        fuel_type,
        nb_steps,
        step_size,
        perimeter_resolution,
        minimal_propagative_front_depth,
        spatial_increment,
        look_ahead_distance_for_time_gradient,
        run_id
    )
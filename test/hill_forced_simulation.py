from forefire_helper import *
from simulation import UniformWindForeFireSimulation
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from forefire_TF_helpers import save_model_structure2
import tensorflow as tf
import pdb


def hill(x, mean, cov, height=100):
    N = len(mean)
    den = (2*np.pi)**(N/2) * np.linalg.det(cov)**0.5
    exp = np.exp(-0.5 * np.einsum('...k,kl,...l->...', x-mean, np.linalg.inv(cov), x-mean))
    gaussian = exp / den
    altitude = height * (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    return altitude


def hill_simulation(config):
    if 'save_exp' in config:
        save_exp = os.path.join(
            config['save_exp'], 
            '{:%Y-%m-%d--%H:%M:%S}'.format(datetime.datetime.now()))
        os.makedirs(save_exp)
    else:
        save_exp = None
        logger_path = None
                            
    propagation_model = config['propagation_model']
    nn_ros_model_path = config['nn_ros_model_path']
    inputs = config['inputs']
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='relu', input_shape=(1,))
    ])
    model_path = os.path.join(save_exp, propagation_model + '.ffann')
    save_model_structure2(model, model_path, inputs, 'ROS')

    if config['logger']:
        logger_path = os.path.join(save_exp, f'{propagation_model}_db.csv')

    if 'fuels_table' in config:
        fuels_table = fuels_table
    else:
        fuels_table = get_fuels_table(propagation_model)
    fuel_type = config['fuel_type']
    domain_width = config['domain_width']
    domain_height = config['domain_height']

    # Create hill with a 2D gaussian
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

    altitude_map = hill(map, mean, cov) 
    # plt.imshow(altitude_map); plt.show()  

    fuel_map = np.ones_like(altitude_map)
    # fuel_map[:, :domain_width // 2] = fuel_type[0]

    fuel_map[np.tril(fuel_map, k=-100) == 0] = fuel_type[0]
    fuel_map[np.tril(fuel_map, k=-100) == 1] = fuel_type[1]
    fuel_map[domain_height // 2 :,:] = fuel_type[1]
    
    horizontal_wind = config['horizontal_wind']
    vertical_wind = config['vertical_wind']

    fire_front = config['fire_front']
    fire_observation = config['fire_observation']

    kwargs = {
        'fire_observation': fire_observation,
        'logger_path': logger_path,
        'model_path': model_path,
    }

    simulation = UniformWindForeFireSimulation(
        "BMapLoggerForANNTraining",
        fuels_table,
        horizontal_wind,
        vertical_wind,
        fuel_map,
        altitude_map,
        fire_front,
        **kwargs
    )

    nb_steps = config['nb_steps']
    step_size = config['step_size']

    # Run simulation
    pathes = simulation(nb_steps, step_size)

    ##-----------------------------------------------------------------------------
    ##   Plot the simulated Fronts
    ##-----------------------------------------------------------------------------
    plotExtents = (
        float(simulation.ff["SWx"]),
        float(simulation.ff["SWx"]) + float(simulation.ff["Lx"]),
        float(simulation.ff["SWy"]),
        float(simulation.ff["SWy"]) + float(simulation.ff["Ly"]))
    

    # Set ForeFire simulation directory
    simulation.ff['caseDirectory'] = '/'.join(save_exp.split('/')[:-1])
    simulation.ff['fireOutputDirectory'] = save_exp.split('/')[-1]
    simulation.ff['experiment'] = 'simulation'
    
    if save_exp:
        # Save simulation contour lines
        plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], plotExtents, None, title=propagation_model, save_exp=save_exp)
        
        # Save config file
        with open(os.path.join(save_exp, 'config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # Save altitude map
        fig = plt.figure()
        plt.imshow(altitude_map)
        plt.colorbar()
        plt.title('Altitude (m)')
        plt.savefig(os.path.join(save_exp, 'altitude.pdf'),  dpi=100, bbox_inches='tight', pad_inches=0.05)

        # Save matrix of burned area and arrival time
        simulation.ff.execute("save[]")
        sim_res = xr.open_dataset(os.path.join(save_exp, simulation.ff['experiment'] + '.0.nc'))
        arrival_time = sim_res.arrival_time_of_front.data

        fig = plt.figure()
        arrival_time[arrival_time < 0] = 0
        plt.imshow(arrival_time)
        plt.colorbar()
        plt.savefig(os.path.join(save_exp, 'arrival_time.pdf'),  dpi=100, bbox_inches='tight', pad_inches=0.05)

    else:
        plot_simulation(pathes, simulation.fuel_map[0, 0], simulation.altitude_map[0, 0], plotExtents, title=propagation_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to the config file (.yaml) of the simulation.')
    parser.add_argument('--propagation_model', type=str, default='RothermelAndrews2018', 
                        help='Rate of Spread model (default: RothermelAndrews2018)')
    parser.add_argument('--fuel_type', type=int, default=6,
                        help='Index of the fuel type to use (default: 6)')
    parser.add_argument('--domain_width', type=int)
    parser.add_argument('--domain_height', type=int)
    parser.add_argument('--horizontal_wind', type=float)
    parser.add_argument('--vertical_wind', type=float)
    parser.add_argument('--slope', type=float)
    parser.add_argument('--nb_steps', type=int,
                        help='Number of simulation steps')
    parser.add_argument('--step_size', type=float,
                        help='Duration (in seconds) between each step.')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
    else:
        config = vars(args)

    hill_simulation(config)
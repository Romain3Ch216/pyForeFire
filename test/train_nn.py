""" Script to train a ROS model emulator
Ex: python train_nn.py --root /home/ai4geo/Documents/nn_ros_models --target_ros_model RothermelAndrews2018 --n_samples 10000 --epochs 200 --overwrite
"""
import os
import sys
import logging
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time

from forefire_TF_helpers import save_to_json

import pdb


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def train(model, dataset, model_path, config):
    train_data, train_target, val_data, val_target = dataset

    l1_reg = tf.keras.regularizers.l1(config['l1_reg_coeff'])
    for layer in model.layers:
        layer.kernel_regularizer = l1_reg 

    # Define SGD algorithm
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    # Compile the model for regression
    model.compile(
        optimizer=optimizer,
        loss=config['loss']
        )
    
    # Stop optimization when validation loss increases
    earlyStopping = EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=0, mode='min')
    
    # Save model if validation loss is improved
    mcp_save = ModelCheckpoint(config['model_path'], save_best_only=True, monitor='val_loss', mode='min')
    
    # Reduce learning rate when validation loss reaches a plateau
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=config['lr_scheduler']['factor'], 
        patience=config['lr_scheduler']['patience'], 
        verbose=1, 
        min_delta=config['lr_scheduler']['min_delta'], 
        mode='min'
        )
    
    # Train the model
    model.fit(
        train_data,
        train_target,
        epochs=config['epochs'], 
        batch_size=config['batch_size'],
        validation_data=(val_data, val_target),
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    model.save(model_path)  # Saving the model

def main(args):
    nn_model_path = os.path.join(args.root, args.nn_model_path)
    
    train_data = pd.read_csv(os.path.join(args.data_path, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(args.data_path, 'val_data.csv'))
    input_names = [
        'fuel.fl1h_tac', 
        'fuel.fd_ft', 
        'fuel.Dme_pc',
        'fuel.SAVcar_ftinv',
        'fuel.H_BTUlb',
        'fuel.totMineral_r',
        'fuel.effectMineral_r',
        'fuel.fuelDens_lbft3',
        'fuel.mdOnDry1h_r',
        'normalWind', 
        'slope'
        ]
    output_names = ['ROS']

    train_target = np.array(train_data[output_names])
    val_target = np.array(val_data[output_names])
    train_data = np.array(train_data[input_names])
    val_data = np.array(val_data[input_names])

    dataset = (train_data, train_target, val_data, val_target)

    train_config = {
        'optimizer': 'adam',
        'loss': 'mean_absolute_error',
        'h_dim': args.h_dim,
        'n_hidden_layers': args.n_hidden_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'l1_reg_coeff': args.l1_reg_coeff,
        'val_prop': args.val_prop,
        'learning_rate': args.lr,
        'patience': args.patience,
        'model_path': nn_model_path,
        'lr_scheduler': {'factor': 0.5, 'patience': 5, 'min_delta': 1e-4}
    }


    #---Model definition---#
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(train_data)

    model_layers = [
        normalization_layer,
        tf.keras.layers.Dense(train_config['h_dim'], activation='relu', input_shape=(train_data.shape[1],))
    ]

    for _i in range(1, train_config['n_hidden_layers']):
        model_layers.append(tf.keras.layers.Dense(train_config['h_dim'], activation='relu'))
    model_layers.append(tf.keras.layers.Dense(1))
                        
    model = tf.keras.Sequential(model_layers)

    # Train the model
    save_to_json(train_config, os.path.join(nn_model_path, 'config.json'))

    if args.overwrite or not os.path.exists(os.path.join(nn_model_path, 'saved_model.pb')):
        logger.info('Optimize neural network')
        stime = time.time()
        train(model, dataset, nn_model_path, train_config)
        ptime = (time.time() - stime) / 60
        logger.info(f'Optimized NN in {ptime:.2f}min')
    else:
        logger.info(f'Neural network has already been trained - params saved in {nn_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Path to your root folder')
    parser.add_argument('--data_path', type=str, help='Path to the data set')
    parser.add_argument('--nn_model_path', type=str, default='dense_net', help='Path to NN')
    parser.add_argument('--n_samples', type=float, default=2**15,
                        help='Number of training data points')
    parser.add_argument('--h_dim', type=int, default=64,
                        help='Number of hidden neurons per layer')
    parser.add_argument('--n_hidden_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for SGD')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--l1_reg_coeff', type=float, default=1e-2,
                        help='Coefficient of L1 norm regularization (enforces sparsity)')
    parser.add_argument('--val_prop', type=float, default=0.2,
                        help='Percentage of training data for validation')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs after which training is stopped if validation loss keeps increasing')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite trained model')
    args = parser.parse_args()    
    main(args)
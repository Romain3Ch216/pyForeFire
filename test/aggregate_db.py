import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb


db_folder = '/home/ai4geo/Documents/nn_ros_models/hill_experiments'

if os.path.exists(os.path.join(db_folder, 'filtered_data.csv')):
    data = pd.read_csv(os.path.join(db_folder, 'filtered_data.csv'))
else:
    data_list = []
    exp_id = 0
    for i, file in enumerate(os.listdir(db_folder)):
        ext = file.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(os.path.join(db_folder, file), delimiter=';')
            print(os.path.join(db_folder, file))
            data_by_node = [data[data.nodeID == node] for node in np.unique(data.nodeID)]
            data_by_node = [x[:-1] for x in data_by_node]
            if len(data_by_node) > 0:
                data = pd.concat(data_by_node, ignore_index=False)
                data['exp'] = exp_id
                data_list.append(data)
                exp_id += 1
    data = pd.concat(data_list, ignore_index=True)
    data = data[data.ROS < 10]
    data.to_csv(os.path.join(db_folder, 'filtered_data.csv'), header=[x for x in data.columns])

pdb.set_trace()

if not os.path.exists(os.path.join(db_folder, 'train_data.csv')):
    # Define validation set
    fuels, val_set = [], []
    i = 0
    for exp_id in np.unique(data.exp):
        fuel = np.unique(data[data.exp==exp_id]['fuel.SAVcar_ftinv'])
        if fuels.count(fuel) < 2:
            val_set.append(exp_id)
            fuels.append(fuel)
            i += 1
    train_set = [x for x in np.unique(data.exp) if x not in val_set]
    val_data = data.loc[data['exp'].isin(val_set)]
    train_data = data.loc[data['exp'].isin(train_set)]
    train_data.to_csv(os.path.join(db_folder, 'train_data.csv'), header=[x for x in data.columns])
    val_data.to_csv(os.path.join(db_folder, 'val_data.csv'), header=[x for x in data.columns])

ros = data.ROS
wind = data.normalWind
slope = data.slope
x = data.nodeLocationX
y = data.nodeLocationY
threshold = 5

def plot_trajectory(data, node_id, exp):
    data = data[data.exp == exp]
    data = data[data.nodeID == node_id]
    x, y = data.nodeLocationX, data.nodeLocationY
    ros = data.ROS
    fig = plt.figure()
    for i in range(0, len(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'bo-')
    plt.scatter(x, y, c=ros)
    plt.show()

def plot_fire_front(data, exp):
    pdb.set_trace()
    data = data[data.exp == exp]
    max_time = data.nodeTime.max()
    

    for t in np.linspace(0, max_time, int(max_time // 10)):
        fig = plt.figure()
        current_data = data[data.nodeTime <= t]
        data_by_node = [current_data[current_data.nodeID == node] for node in np.unique(current_data.nodeID)]
        data_by_node = [x[-1:] for x in data_by_node]
        pos = np.array([[x.nodeLocationX, x.nodeLocationY] for x in data_by_node])
        for i in range(0, len(pos)):
            # plt.plot(pos[i:i+2, 0, 0],pos[i:i+2, 1, 0], 'bo-')
            plt.scatter(pos[i, 0, 0], pos[i, 1, 0], color='black')
        plt.show()


# for node_id in [8, 40, 490, 492]:
# plot_fire_front(data, exp=1)

# pdb.set_trace()

# fig = plt.figure()
# plt.scatter(data.ROS, data.nodeTime)
# plt.show()

# fig = plt.figure()
# plt.scatter(data[data.ROS > 5].ROS, data[data.ROS > 5].nodeTime)
# plt.title('nodeTime vs ROS for ROS > 5 m/s')
# xx = np.linspace(5, 10, 100)
# y1 = np.zeros_like(xx)
# y2 = data.nodeTime.max() * np.ones_like(xx)
# plt.plot(xx, y1, label='nodeTilme = 0s')
# plt.plot(xx, y2, label='nodeTime = nodeTime.max()')
# plt.legend()
# plt.xlabel('ROS (m/s)')
# plt.ylabel('nodeTime (s)')
# plt.show()


# mask = (wind <= 3) * (wind >= 2)
mask = np.ones_like(wind).astype(bool)
# mask = np.ones_like(wind)

small_mask = ros <= threshold
small_ros = ros[small_mask]
small_wind = wind[small_mask]
small_slope = slope[small_mask]
small_x = x[small_mask]
small_y = y[small_mask]

high_mask = ros > threshold
high_ros = ros[high_mask]
high_wind = wind[high_mask]
high_slope = slope[high_mask]
high_x = x[high_mask]
high_y = y[high_mask]


# bmap = np.zeros((1000, 1000))
# bmap[np.array(small_y).astype(int), np.array(small_x).astype(int)] = -1
# bmap[np.array(high_y).astype(int), np.array(high_x).astype(int)] = 1

# plt.imshow(bmap)
# plt.colorbar()
# plt.show()
# pdb.set_trace()
   
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data.normalWind[mask], data.slope[mask], data.ROS[mask], s=1, color='black')
# # ax.scatter(small_wind, small_slope, small_ros, s=1, color='black')
# # ax.scatter(high_wind, high_slope, high_ros, s=1, color='red')
# ax.set_xlabel('Normal wind (m/s)')
# ax.set_ylabel('Tan(slope)')
# ax.set_zlabel('ROS (m/s)')
# plt.show()


# fig, ax = plt.subplots(2)
# ax[0].scatter(small_wind, small_ros, color='#731963', s=1)
# ax[0].set_xlabel('Normal wind (m/s)')
# ax[0].set_ylabel('ROS (m/s)')
# ax[1].scatter(small_slope, small_ros, color='#0B3C49', s=1)
# ax[1].scatter(high_slope, high_ros, color)
# ax[1].set_yscale('log')
# ax[1].set_xlabel('Tan(slope)')
# ax[1].set_ylabel('ROS (m/s)')
# plt.show()

fig, ax = plt.subplots(2)
ax[0].scatter(data.normalWind[mask], data.ROS[mask], color='#731963', s=1)
ax[0].set_xlabel('Normal wind (m/s)')
ax[0].set_ylabel('ROS (m/s)')
ax[1].scatter(data.slope[mask], data.ROS[mask], color='#0B3C49', s=1)
# ax[1].set_yscale('log')
# for slope, ros, x, y in zip(high_slope, high_ros, high_x, high_y):
#      plt.text(slope, ros, '({}, {})'.format(x, y))
ax[1].set_xlabel('Tan(slope)')
ax[1].set_ylabel('ROS (m/s)')
plt.show()

# fig = plt.figure()
# plt.imshow(data.corr())
# plt.title("Correlation matrix")
# plt.show()
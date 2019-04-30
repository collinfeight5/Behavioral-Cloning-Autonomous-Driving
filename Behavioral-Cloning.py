# Behavorial Cloning Script
# Written By Collin Feight


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda


data_path = 'data/'
drive_log = pd.read_csv(data_path+'driving_log.csv')

# Only use 20% of Data when steering is 0, to prevent over-fitting
drive_log = drive_log[drive_log['steering'] != 0].append(drive_log[drive_log['steering'] == 0].sample(frac=0.2))


def get_data(line):
    data_dict = {}
    # center image
    center_img = io.imread(data_path + line['center'].strip())
    center_ang = line['steering']
    data_dict['center'] = (center_img, center_ang)

    # flip image if steering is not 0
    if line['steering']:
        flip_img = center_img[:, ::-1]
        flip_ang = center_ang * -1
        data_dict['flip'] = (flip_img, flip_ang)

    # left image
    left_img = io.imread(data_path + line['left'].strip())
    left_ang = center_ang + .205 * np.random.random()
    left_ang = min(left_ang, 1)
    data_dict['left_camera'] = (left_img, left_ang)

    # right image
    right_img = io.imread(data_path + line['right'].strip())
    right_ang = center_ang - .205 * np.random.random()
    right_ang = max(right_ang, -1)
    data_dict['right_camera'] = (right_img, right_ang)

    return data_dict

# Gather Data and structure in x_train, y_train form for images and angle correction respectively
x_train = []
y_train = []
for i, row in drive_log.iterrows():
    data = get_data(row)
    for image, angle in data.values():
        x_train.append(image)
        y_train.append(angle)

x_train = shuffle(np.array(x_train))
y_train = shuffle(np.array(y_train))

batch = 128
epochs = 7
activation_type = 'elu'
dropout = .3

def train_model(X_train, Y_train):
    model = Sequential()
    # Crop
    # Max pooling causes data to crash?
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255) - .5))
    # Start Convolutional NN
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Dropout(dropout))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation=activation_type))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation=activation_type))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1164, activation=activation_type))
    model.add(Dense(100, activation=activation_type))
    model.add(Dense(50, activation=activation_type))
    model.add(Dense(10, activation=activation_type))
    model.add(Dense(1))

    print('Training the model')

    model.compile(optimizer='adam', loss='mse')
    # Refresh weights every time new training happens
    model.save_weights('pre_saved_weights.h5')
    model.load_weights('pre_saved_weights.h5')
    # Assign Data
    model.fit(X_train, Y_train, batch_size=batch, epochs=epochs, validation_split=.2)
    model.save('model_preprocessed.h5')
    return None


train_model(x_train, y_train)

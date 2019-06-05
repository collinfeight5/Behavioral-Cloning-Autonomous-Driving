#### Goal for this project was to create/train a CNN that would enable a car to autonomously drive in a simulator
#### The approach that was used was to gather data from operating the car in the simulator, and use that data to train the CNN
#### Written by Collin Feight

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda

# Data_path changes per new training folder being used
data_path = 'data/'
drive_log = pd.read_csv(data_path+'driving_log.csv')
# Below for plotting purposes
# plt.hist(drive_log['steering'], bins=100)
# plt.title(r'Steering Angle Values')
# plt.xlabel('Steering Angle Degree')
# plt.ylabel('Count')
# plt.show()

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
    left_ang = center_ang + .205 #* np.random.random()
    left_ang = min(left_ang, 1)
    data_dict['left_camera'] = (left_img, left_ang)

    # right image
    right_img = io.imread(data_path + line['right'].strip())
    right_ang = center_ang - .205 # * np.random.random()
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

x_train = np.array(x_train)
y_train = np.array(y_train)

batch_size = 120
epochs = 7
activation_type = 'elu'
dropout = .3


# Max pooling causes data to crash?
def train_model(X_train, Y_train):
    model = Sequential()
    # Crop
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255) - .5))
    # Start Convolutional NN
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type))
    model.add(Conv2D(64, (3, 3), activation=activation_type))
    model.add(Conv2D(64, (3, 3), activation=activation_type))
    model.add(Flatten())
	model.add(Dropout(dropout))
    model.add(Dense(1164, activation=activation_type))
    model.add(Dense(100, activation=activation_type))
    model.add(Dense(50, activation=activation_type))
    model.add(Dense(10, activation=activation_type))
    model.add(Dense(1))
    print('Training the model')
    model.compile(optimizer='adam', loss='mse')
    #model.reset_states()
    # train model using additional data gathered in simulator
    #model.load_weights('model_preprocessed.h5')
    # Assign Data
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=.2, shuffle=True)
    model.save('model_preprocessed.h5')
    return None

# Function call to train model given input data
train_model(x_train, y_train)

# Used for plotting training/validation results
#history = train_model(x_train, y_train)
#print(history.history.keys())
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model accuracy')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()


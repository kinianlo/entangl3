# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 00:56:59 2019

@author: Kin Ian Lo
"""

import pandas as pd
import numpy as np
import utilities
from keras.models import Sequential
from keras.layers import Dense, Activation
from time import time
import data_generation

df_raw = pd.read_csv('dataset/10k10k_m10_100_1000.csv', index_col=0)

nb_measurement = 1000
distribution = 'bloch_sphere_uniform'
#distribution = 'entropy_uniform'

usage = 'train'
df = df_raw.loc[(df_raw['nb_measurement']==nb_measurement) &
                (df_raw['distribution']==distribution) &
                (df_raw['usage']==usage)]


bloch_vectors = df[['bloch_vector_1',
                    'bloch_vector_2',
                    'bloch_vector_3']].values
bloch_lengths = np.sqrt(np.sum(bloch_vectors**2, axis=1))
VN_entropies = utilities.get_VN_entropy(bloch_lengths)

nb_positive_outcomes = df[['nb_positive_outcome_1',
                           'nb_positive_outcome_2',
                           'nb_positive_outcome_3']].values

measured_bloch_vectors = 2*nb_positive_outcomes/nb_measurement - 1

x_train = measured_bloch_vectors
y_train = VN_entropies/np.log(2)

usage = 'test'
df = df_raw.loc[(df_raw['nb_measurement']==nb_measurement) &
                (df_raw['distribution']==distribution) &
                (df_raw['usage']==usage)]


bloch_vectors = df[['bloch_vector_1',
                    'bloch_vector_2',
                    'bloch_vector_3']].values
bloch_lengths = np.sqrt(np.sum(bloch_vectors**2, axis=1))
VN_entropies = utilities.get_VN_entropy(bloch_lengths)

nb_positive_outcomes = df[['nb_positive_outcome_1',
                           'nb_positive_outcome_2',
                           'nb_positive_outcome_3']].values

measured_bloch_vectors = 2*nb_positive_outcomes/nb_measurement - 1

x_test = measured_bloch_vectors
y_test = VN_entropies/np.log(2)


best_model = None
best_RMSE = None
param = ''

for initializer in ['normal', 'he_normal']:
    for optimizer in ['adam', 'adadelta']:
        for nb_neurons in [64, 128, 256, 512, 1024]:
            for i in range(3):
                batch_size = 100
                nb_epoch = 300

                input_neurons = 3
                output_neurons = 1
                output_activation = 'linear'

                nb_hl = 1

                hl_neurons = [nb_neurons] * nb_hl
                hl_activations = ['relu'] * nb_hl

                model = Sequential()
                model.add(Dense(units=hl_neurons[0], input_dim=input_neurons, kernel_initializer=initializer))
                model.add(Activation(hl_activations[0]))

                for l in range(1, len(hl_neurons)):
                    model.add(Dense(units=hl_neurons[l], input_dim=hl_neurons[l-1], kernel_initializer=initializer))
                    model.add(Activation(hl_activations[l]))

                model.add(Dense(units=output_neurons, input_dim=hl_neurons[-1], kernel_initializer=initializer))
                model.add(Activation(output_activation))

                model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
                model.summary()

                model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=0)
                t1 = time()
                y_test_pred = np.squeeze(model.predict(x_test))
                t2 = time()
                print('time used to predict: {:}'.format(t2-t1))
                error = y_test_pred - y_test
                RMSE = np.sqrt(np.mean(error**2))

                if best_RMSE == None or RMSE < best_RMSE:
                    best_RMSE = RMSE
                    best_model = model
                    param = '{:}m {:}hl init={:} optim={:} nb_neurons={:}'.format(nb_measurement, nb_hl, initializer, optimizer, nb_neurons)

print(param)
#print('RMSE: {:}'.format(best_RMSE))

SDI_RMSE_list = []
RMSE_list = []
for i in range(10):
    df_raw = data_generation.simulate_measurements(1000000, nb_measurement, 'bloch_sphere_uniform', 'test')
    usage = 'test'
    df = df_raw.loc[(df_raw['nb_measurement']==nb_measurement) &
                    (df_raw['distribution']==distribution) &
                    (df_raw['usage']==usage)]


    bloch_vectors = df[['bloch_vector_1',
                        'bloch_vector_2',
                        'bloch_vector_3']].values
    bloch_lengths = np.sqrt(np.sum(bloch_vectors**2, axis=1))
    VN_entropies = utilities.get_VN_entropy(bloch_lengths)

    nb_positive_outcomes = df[['nb_positive_outcome_1',
                               'nb_positive_outcome_2',
                               'nb_positive_outcome_3']].values

    measured_bloch_vectors = 2*nb_positive_outcomes/nb_measurement - 1

    x_test = measured_bloch_vectors
    y_test = VN_entropies/np.log(2)
    y_test_pred = np.squeeze(model.predict(x_test))
    error = y_test_pred - y_test
    RMSE_list.append(np.sqrt(np.mean(error**2)))

    bl = np.clip(np.sqrt(np.sum(x_test**2, axis=1)), 0, 1)
    y_SDI = utilities.get_VN_entropy(bl)/np.log(2)
    SDI_error = y_SDI - y_test
    SDI_RMSE_list.append(np.sqrt(np.mean(SDI_error**2)))


print('RMSE = {:}+-{:}'.format(np.mean(RMSE_list), np.std(RMSE_list)))
print('SDI_RMSE = {:}+-{:}'.format(np.mean(SDI_RMSE_list), np.std(SDI_RMSE_list)))

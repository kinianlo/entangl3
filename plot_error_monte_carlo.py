# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:28:37 2019

@author: Kin Ian Lo
"""

import matplotlib.pyplot as plt
import numpy as np
import data_generation
import utilities

ANN_N_range = [1, 2, 3]
ANN_RMSE = [0.21794391146906844, 0.1086767149861523, 0.041838265737885294]
SDI_N_range = np.log(np.array(np.logspace(1, 3, 20), dtype='int'))/np.log(10)
SDI_RMSE = []

distribution = 'bloch_sphere_uniform'
for N_raw in SDI_N_range:
    nb_measurement = int(np.round(10**N_raw))
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

    bl = np.clip(np.sqrt(np.sum(x_test**2, axis=1)), 0, 1)
    y_SDI = utilities.get_VN_entropy(bl)/np.log(2)
    SDI_error = y_SDI - y_test
    SDI_RMSE.append(np.sqrt(np.mean(SDI_error**2)))

plt.figure()
plt.plot(SDI_N_range, SDI_RMSE, 'mx-', label='SDI')
plt.plot(ANN_N_range, ANN_RMSE, 'kx:', label='ANN')
plt.xlim([None, None])
plt.ylim([0, None])
plt.xlabel('$log_{10}(N)$')
plt.legend()

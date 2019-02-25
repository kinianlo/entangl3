# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:44:00 2018

@author: Kin Ian Lo
"""
import os
import pandas as pd
import numpy as np
import utilities

dataset_folder = 'dataset'

def simulate_experiment(nb_sample, nb_measurement=np.inf,
                          distribution='bloch_sphere_uniform',
                          usage='train'):
    """
    Simulate a set of experimental outcomes. 
    First nb_sample of Bloch vectors are sampled according to the given
    distribution. For each sampled Bloch vector, nb_measurement spin-1/2 
    measurements are done for each of the x, y and directions. 
    
    Args:
        nb_sample: number of samples of Bloch vector
        nb_measurement: number of measurements made on each x, y and z axis 
        distribution: the distribtuion of Bloch vectors sampled
        usage: can be 'train' or 'test' (for bookkepping only)
        
    Returns:
        A panda dataframe with the following columns:
            nb_measurement: (see above)
            distribution: (see above)
            usage: (see above)
            bloch_vector_1: x-component of the Bloch vector 
            bloch_vector_2: y-component of the Bloch vector 
            bloch_vector_3: z-component of the Bloch vector 
            nb_positive_outcome_1: no. of 'up-state' sigma_x measurement outcomes
            nb_positive_outcome_2: no. of 'up-state' sigma_y measurement outcomes
            nb_positive_outcome_3: no. of 'up-state' sigma_z measurement outcomes      
        There are nb_sample rows and each row represent an experimental trial
    """
    
    # uniform sampling of 3-d unit vectors
    unit_vectors = np.random.normal(size=(nb_sample, 3))
    # normalisation
    unit_vectors = unit_vectors/np.sqrt(np.sum(unit_vectors**2, axis=1, keepdims=True))

    # sampling of vector lengths
    bloch_lengths = None
    if distribution == 'bloch_sphere_uniform':
        bloch_lengths = (np.random.uniform(size=(nb_sample, 1)))**(1/3)
    elif distribution == 'entropy_uniform':
        bloch_lengths = utilities.get_bloch_length(np.random.uniform(size=(nb_sample, 1))*np.log(2))

    # scale the unit vectors by the sampled bloch length
    bloch_vectors = bloch_lengths*unit_vectors

    # simulate experimental outcome with binomial distributions
    nb_positive_outcome = np.zeros_like(bloch_vectors, dtype=np.int)
    if np.isfinite(nb_measurement):
        for i in range(nb_sample):
            for j in range(3):
                nb_positive_outcome[i,j] = np.random.binomial(nb_measurement, (bloch_vectors[i,j]+1)/2)

    ## plot measured bloch lengths for checking
    #N = np.sum((2*nb_positive_outcome-nb_measurement)**2, axis=1, keepdims=True)
    #measured_bloch_lengths = np.sqrt(N)/nb_measurement
    #plt.figure()
    #plt.scatter(measured_bloch_lengths, utilities.get_VN_entropy(bloch_lengths), s=5)
    #plt.scatter(measured_bloch_lengths, utilities.get_VN_entropy(measured_bloch_lengths), s=5)

    df = pd.DataFrame({'nb_measurement': nb_measurement,
                       'distribution': distribution,
                       'usage': usage,
                       'bloch_vector_1': bloch_vectors[:,0],
                       'bloch_vector_2': bloch_vectors[:,1],
                       'bloch_vector_3': bloch_vectors[:,2],
                       'nb_positive_outcome_1': nb_positive_outcome[:,0],
                       'nb_positive_outcome_2': nb_positive_outcome[:,1],
                       'nb_positive_outcome_3': nb_positive_outcome[:,2]
                       })
    return df

if __name__ == '__main__':
    df = pd.concat([simulate_experiment(10000, 5, 'bloch_sphere_uniform', 'train'),
                    simulate_experiment(10000, 5, 'bloch_sphere_uniform', 'test'),
                    simulate_experiment(10000, 10, 'bloch_sphere_uniform', 'train'),
                    simulate_experiment(10000, 10, 'bloch_sphere_uniform', 'test'),
                    simulate_experiment(10000, 20, 'bloch_sphere_uniform', 'train'),
                    simulate_experiment(10000, 20, 'bloch_sphere_uniform', 'test'),
                    simulate_experiment(10000, 40, 'bloch_sphere_uniform', 'train'),
                    simulate_experiment(10000, 40, 'bloch_sphere_uniform', 'test')
                    ], ignore_index=True)
    df.to_csv(os.path.join(dataset_folder, '10k10k_m5_10_20_40.csv'))




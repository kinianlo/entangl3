# Entanglement Estimation 
This repository includes all the code used in the completetion of the BSc project.  See the report (BSc_Project_Report.pdf) for details.

### Prerequisites
You will need the following python packages to run the code in this repository.
```
Numpy
Scipy
Pandas 
Matplotlib
Keras
Tensorflow
```

## List of Files 
* __data_generation.py__ - Generates simulated experimental data used for training neural networks and monte carlo methods. Data are saved as pandas dataframe and can be export to csv. 
* __prob_integral.py__ - Perform the integrations for the Bayesian methods. 
* __utilities.py__ - Helper functions.
* __hyper_opt_exact.py__ - Hyperparameter optimisations where errors are calculated using the Bayesian methods.
* __hyper_opt_monte_carlo.py__ - Hyperparameter optimisations where errors are calculated using the testing data generated with __data_generation.py__. 
* __plot_error_exact.py__ - Summarises the root mean squared error (RMSE) for SDI, BME, and the maximum possible RMSE.
* __plot_error_monte_carlo.py__ - Summarises the root mean squared error (RMSE) for SDI using data generated with __data_generation.py__. 
* __BSc_Project_Report.pdf__ - Includes everything you need to know about the project. 
* __dataset/10k10k_m5_10_20_40.csv__ - Dataset for N (no. of measurement) = 5, 10, 20 and 40. For each N, there are 10,000 training data and 10,000 testing data. 
* __dataset/10k10k_m10_100_1000.csv__ - Dataset for N (no. of measurement) = 10, 100 and 1000. For each N, there are 10,000 training data and 10,000 testing data. 
* __dataset/10k2k_m5_to_m1000.csv__ - Dataset for N (no. of measurement) = 5, 10, 50, 100, 500 and 1000. For each N, there are 10,000 training data and 10,000 testing data. 
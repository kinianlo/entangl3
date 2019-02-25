# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:35:00 2019

@author: Kin Ian Lo
"""
import matplotlib.pyplot as plt
from prob_integral import VNE_CoM_Predictor

ann_rmse = [0.24090482614923364, 0.21746049030261544, 0.1854438836281621, 0.14984349360520102]
N_range_ann = [5, 10, 20, 40]

qst_rmse = []
min_rmse = []
max_rmse = []
N_range = list(range(1, 40))
#N_range = list(range(1, 20)) + list(range(20, 40, 4))

for N in N_range:
    pred = VNE_CoM_Predictor(N)
    qst_rmse.append(pred.get_anal_RMSE()[0])
    min_rmse.append(pred.get_min_RMSE()[0])
    max_rmse.append(pred.get_max_RMSE()[0])
    print('N={N} Done'.format(N=N))

plt.figure()
plt.plot(N_range, max_rmse, 'g', alpha=0.7, label='MAX')
plt.plot(N_range, qst_rmse, 'm-', label='SDI')
plt.plot(N_range_ann, ann_rmse, 'wx')
plt.plot(N_range, min_rmse, 'c-', label='BME(MIN)')
plt.plot(N_range_ann, ann_rmse, 'kx', label='ANN')
plt.ylabel('RMS error')
plt.xlabel(r'$N$')
plt.xlim([0, 50])
plt.ylim([0, None])
plt.legend()
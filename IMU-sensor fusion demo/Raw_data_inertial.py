# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:09:30 2015

@author: mchijtz4(tahmina.zebin@manchester.ac.uk)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=pd.read_csv('F:\\Supporting files\\IMU-sensor fusion demo\\sensor.csv')
statistical_metrics=dataset.iloc[:,2:32].describe()

pelvis=dataset.iloc[:,2:8].describe()
Rt=dataset.iloc[:,8:14].describe()
lt=dataset.iloc[:,14:20].describe()
Rs=dataset.iloc[:,20:26].describe()
ls=dataset.iloc[:,26:32].describe()


g = sns.pairplot(dataset, vars=["Pelvis_Acc_X", "Pelvis_Acc_Y", "Pelvis_Acc_Z"])
g1 = sns.pairplot(dataset, vars=["Pelvis_Gyr_X", "Pelvis_Gyr_Y", "Pelvis_Gyr_Z"])

g3=sns.pairplot(dataset, vars=["LT_Acc_X", "LT_Acc_Y", "LT_Acc_Z"])
g4 = sns.pairplot(dataset, vars=["LT_Gyr_X", "LT_Gyr_Y", "LT_Gyr_Z"])
g5=sns.pairplot(dataset, vars=["RT_Acc_X", "RT_Acc_Y", "RT_Acc_Z"])
g6 = sns.pairplot(dataset, vars=["RT_Gyr_X", "RT_Gyr_Y", "RT_Gyr_Z"])
g7=sns.pairplot(dataset, vars=["LS_Acc_X", "LS_Acc_Y", "LS_Acc_Z"])
g8 = sns.pairplot(dataset, vars=["LS_Gyr_X", "LS_Gyr_Y", "LS_Gyr_Z"])
g9=sns.pairplot(dataset, vars=["RS_Acc_X", "RS_Acc_Y", "RS_Acc_Z"])
g10 = sns.pairplot(dataset, vars=["RS_Gyr_X", "RS_Gyr_Y", "RS_Gyr_Z"])

g11=sns.pairplot(dataset, vars=["Pelvis_Acc_Z", "LT_Acc_Z", "LS_Acc_Z"],kind='reg')

g12=sns.pairplot(dataset, vars=["Pelvis_Acc_Z", "RT_Acc_Z", "RS_Acc_Z"],kind='reg')

g11=sns.pairplot(dataset, vars=["Pelvis_Gyr_X", "LT_Gyr_X", "LS_Gyr_X"],kind='reg')

g12=sns.pairplot(dataset, vars=["Pelvis_Gyr_X", "RT_Gyr_X", "RS_Gyr_X"],kind='reg')
sns.pairplot(dataset, vars=["LT_Acc_Z", "RT_Acc_Z"],kind='reg')
sns.pairplot(dataset, vars=["LS_Acc_Z", "RS_Acc_Z"],kind='reg')

sns.set(color_codes=True)
#np.random.seed(sum(map(ord, "regression")))
sns.regplot(x="LT_Acc_Z",y="RT_Acc_Z")

plt.show()

#SNS boxplot

                 
sns.boxplot(dataset);
sns.boxplot(dataset.iloc[:,2:5]);
sns.boxplot(dataset.iloc[:,5:8]);
sns.boxplot(dataset.iloc[:,26:32]);
sns.boxplot(dataset.iloc[:,20:26]);
sns.boxplot(dataset.iloc[:,14:20]);
sns.boxplot(dataset.iloc[:,8:14]);

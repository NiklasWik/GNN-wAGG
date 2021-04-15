#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:57:48 2021

@author: juliasolhed
"""

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)

import pandas as pd 
from matplotlib import pyplot as plt
dataTrain = pd.read_csv("trainPattern.csv")
dataTest = pd.read_csv("testPattern.csv")

#sns.catplot(x="GNN",y="Train_accuracy",hue='A',kind="bar", palette="pastel", edgecolor=".6",data=da)
#sns.catplot(x="GNN", y="AA", hue="A", kind="bar", data=da)

colors = ["#3498DB","#e74c3c","#34495e","#2ecc71"]
sns.set_palette(colors)
g = sns.catplot(x="GNN", y="Test_Acc", hue="Agg", kind="bar", data=dataTest)
g.set(ylim=(85, 86))
plt.title("Test accuracy (PATTERN)")
plt.show(g)

g = sns.catplot(x="GNN", y="Train_Acc", hue="Agg", kind="bar", data=dataTrain)
g.set(ylim=(85.5, 87))
plt.title("Test accuracy (PATTERN)")
plt.show(g)

#g= sns.catplot(x="GNN", y="Train", hue="Agg", kind="swarm",aspect=1, data=da, palette="Spectral")

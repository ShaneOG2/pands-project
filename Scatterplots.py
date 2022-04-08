# Outputs scatter plot of each pair of variables

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

sns.set(style="darkgrid")



def scatterplotDistribution():
    sns.pairplot(df, kind="scatter", hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()
    
    # right: you can give other arguments with plot_kws.
    sns.pairplot(df, kind="scatter", hue="Species", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
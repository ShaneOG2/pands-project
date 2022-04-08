# Save histograms of each variable to png file

from turtle import position
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns

##### Importing data into dataframe

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

def histogramTotal():
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.histplot(data=df, x="SepalLengthCm", kde=True, color="skyblue", ax=axs[0, 0])
    sns.histplot(data=df, x="SepalWidthCm", kde=True, color="olive", ax=axs[0, 1])
    sns.histplot(data=df, x="PetalLengthCm", kde=True, color="gold", ax=axs[1, 0])
    sns.histplot(data=df, x="PetalWidthCm", kde=True, color="teal", ax=axs[1, 1])
    plt.show()
    plt.savefig()

def histogramSpecies():
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.histplot(data=df, x="SepalLengthCm", kde=True, ax=axs[0, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="SepalWidthCm", kde=True, ax=axs[0, 1], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalLengthCm", kde=True, ax=axs[1, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalWidthCm", kde=True, ax=axs[1, 1], hue="Species", palette="Set2")
    plt.show()
    plt.savefig()

histogramSpecies()

# References 
# Plotting histograms: https://www.youtube.com/watch?v=snkkKrek7TU&ab_channel=DataCamp
# Font size: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
# https://python-graph-gallery.com/25-histogram-with-several-variables-seaborn
# https://seaborn.pydata.org/generated/seaborn.histplot.html




# Author: Shane O'Gorman
# Analysis of Iris Data set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import csv

##### Importing data into dataframe

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

##### Writes to summary file

def summaryFile():

    head = df.head() # First 6 rows
    shape = df.shape # Number of rows and collumns
    info = df.info() # Information of df 
    description = df.describe().round(1) 
    numberOfSpecies = df["Species"].value_counts()

    SepalLengthBySpecies = df.groupby("Species")["SepalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    SepalWidthBySpecies = df.groupby("Species")["SepalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    PetalLengthBySpecies = df.groupby("Species")["PetalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    PetalWidthBySpecies = df.groupby("Species")["PetalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])

    with open("Summary.txt", "w") as f:
            
        f.write(("Data Summary\n\n"))
        f.write(("First 6 rows of our data:\n\n")+(str(head)+('\n\n')))
        f.write(("Number of rows and collums:\n\n")+(str(shape)+('\n\n')))
        f.write(("Data types for each variable:\n\n")+(str(info)+('\n\n')))
        f.write(("Overview of statistics:\n\n")+(str(description)+('\n\n')))
        f.write(("Number of species types:\n\n")+(str(numberOfSpecies)+('\n\n')))
        f.write(("Sepal Length:\n\n")+(str(SepalLengthBySpecies)+('\n\n')))
        f.write(("Sepal Width:\n\n")+(str(SepalWidthBySpecies)+('\n\n')))
        f.write(("Petal Length:\n\n")+(str(PetalLengthBySpecies)+('\n\n')))
        f.write(("Petal Width:\n\n")+(str(PetalWidthBySpecies)+('\n\n')))

##### Plots Histograms

def plotHistograms():
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.histplot(data=df, x="SepalLengthCm", kde=True, ax=axs[0, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="SepalWidthCm", kde=True, ax=axs[0, 1], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalLengthCm", kde=True, ax=axs[1, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalWidthCm", kde=True, ax=axs[1, 1], hue="Species", palette="Set2")
    plt.show()
    #plt.savefig()

def plotScatterplots():
    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="SepalLengthCm", y="SepalWidthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="SepalLengthCm", y="PetalLengthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="SepalLengthCm", y="PetalWidthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="SepalWidthCm", y="PetalLengthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="SepalWidthCm", y="PetalLengthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="PetalLengthCm", y="SepalWidthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

def pairPlots():
    sns.pairplot(df, kind="scatter", hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

###### Function calls
#summaryFile()
#plotHistograms()
plotScatterplots()
#pairPlots()



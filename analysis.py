# Author: Shane O'Gorman
# Analysis of Iris Data set

import pandas as pd
import numpy as np
import matplotlib as plt 
import seaborn as sns
import csv

##### Importing data into dataframe

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

##### Explore our data 

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



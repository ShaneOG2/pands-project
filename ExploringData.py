
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns

#### Explore our data 

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

#print(df.head()) # Prints first 6 rows
#print(df.info()) # Prints info about df
#print(df.shape) # Prints shape of df
#print(df.describe()) # Prints a description of df 

#filename = "summary.txt"

# Opens the summary file to write into it
#summary = open(filename, 'w')

# summary.write(df.head())
#head = df.head()
#print(head)

#summary.close() # Close the file when we’re done

SepalLengthBySpecies = df.groupby("Species")["SepalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
print("Sepal Length\n", SepalLengthBySpecies, "\n")

SepalWidthBySpecies = df.groupby("Species")["SepalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
print("Sepal Width\n", SepalWidthBySpecies, "\n")

PetalLengthBySpecies = df.groupby("Species")["PetalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
print("Petal Length\n", PetalLengthBySpecies, "\n")

PetalWidthBySpecies = df.groupby("Species")["PetalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
print("Petal Width\n", PetalWidthBySpecies, "\n")

#print(df)

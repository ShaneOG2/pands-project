
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
#
#print(df.values) # Prints values of df 
#print(df.columns) # Prints collumns of df

#filename = "summary.txt"

# Opens the summary file to write into it
#summary = open(filename, 'w')

# summary.write(df.head())
#head = df.head()
#print(head)

#summary.close() # Close the file when we’re done

SepalBySpecies = df.groupby("Species")["SepalLengthCm","SepalWidthCm"].agg([min, max, np.mean, np.median])
#print(SepalBySpecies)

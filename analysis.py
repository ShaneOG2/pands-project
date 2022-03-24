import pandas as pd
import numpy as np
import os
import matplotlib as plt 
import seaborn as sns

df = pd.read_csv('iris.data')
df.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

print(df.head()) # Prints first 6 rows
#print(df.info()) # Prints info about df
#print(df.shape) # Prints shape of df
print(df.describe()) # Prints a description of df 

#print(df.values) # Prints values of df 
#print(df.columns) # Prints collumns of df

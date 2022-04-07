# Save histograms of each variable to png file

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns

##### Importing data into dataframe

collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

sns.set() # Uses sns which looks better
df[["SepalLengthCm", "SepalWidthCm"]].hist()
#plt.hist(df[["SepalLengthCm", "SepalWidthCm"]]) # Create histogram of Sepal Length with 20 count bins and purple bars
#plt.xlabel("Sepal Length (cm)") # Creates x label
#plt.ylabel("Frequecy") # Creates y label
#plt.title("Histogram of Sepal Length (cm)", fontsize=20) # # Creates title with bigger font size
plt.show()
#plt.savefig("SepalLengthHist.png")

# References 
# Plotting histograms: https://www.youtube.com/watch?v=snkkKrek7TU&ab_channel=DataCamp
# Font size: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib



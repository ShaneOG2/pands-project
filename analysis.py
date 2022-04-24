# Author: Shane O'Gorman
# Analysis of Iris Data set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def main():
    ##### Importing data into dataframe
    collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
    df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

    ##### Function calls
    #summaryFile()
    #plotBoxPlots()
    #plotHistograms()
    #pairPlots()
    #plotScatterplots()
    #corrMatrixPlot()

    ##### Use df2 for machine learning work setting species to numerical values
    df2 = df
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df2["Species"] = le.fit_transform(df2["Species"])

    X = df2.drop(["Species"], axis=1)
    Y = df2["Species"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    


##### Writes to summary file

def summaryFile():

    head = df.head() 
    shape = df.shape 
    datatypes = df.dtypes
    nullcounts = df.isnull().sum()
    numberOfSpecies = df["Species"].value_counts()

    description = df.describe().round(1) 
    
    SepalLengthBySpecies = df.groupby("Species")["SepalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    SepalWidthBySpecies = df.groupby("Species")["SepalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    PetalLengthBySpecies = df.groupby("Species")["PetalLengthCm"].agg([np.mean, np.std, np.min, np.median, np.max])
    PetalWidthBySpecies = df.groupby("Species")["PetalWidthCm"].agg([np.mean, np.std, np.min, np.median, np.max])

    corrMatrix = df.corr()

# Opens text file named summary.txt to write to as f
    with open("Summary.txt", "w") as f:
        
        # Writes to Summary.txt using variable above and converting them to strings using ste
        f.write(("Data Summary\n\n"))
        f.write(("First 6 rows of our data:\n\n")+(str(head)+('\n\n')))
        f.write(("Number of rows and collums:\n\n")+(str(shape)+('\n\n')))
        f.write(("Data types for each variable:\n\n")+(str(datatypes))+('\n\n'))
        f.write(("Null counts:\n\n")+(str(nullcounts))+('\n\n'))
        f.write(("Number of species types:\n\n")+(str(numberOfSpecies)+('\n\n')))

        f.write(("Overview of statistics:\n\n")+(str(description)+('\n\n')))

        f.write(("Sepal Length:\n\n")+(str(SepalLengthBySpecies)+('\n\n')))
        f.write(("Sepal Width:\n\n")+(str(SepalWidthBySpecies)+('\n\n')))
        f.write(("Petal Length:\n\n")+(str(PetalLengthBySpecies)+('\n\n')))
        f.write(("Petal Width:\n\n")+(str(PetalWidthBySpecies)+('\n\n')))

        f.write(("Correlation Matrix:\n\n")+(str(corrMatrix)+('\n\n')))

##### Plots Boxplots

def plotBoxPlots():
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.boxplot(data=df, x="Species", y="SepalLengthCm", ax=axs[0, 0], palette="Set2")
    sns.boxplot(data=df, x="Species", y="SepalWidthCm", ax=axs[0, 1], palette="Set2")
    sns.boxplot(data=df, x="Species", y="PetalLengthCm", ax=axs[1, 0], palette="Set2")
    sns.boxplot(data=df, x="Species", y="SepalWidthCm", ax=axs[1, 1], palette="Set2")
    plt.show()

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

##### Plots Pairplots

def pairPlots():
    sns.pairplot(df, kind="scatter", hue="Species", markers=["o", "s", "D"], palette="Set2")
    plt.show()

##### Plots Scatterplots

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

##### Plots Correlation Matrix

def corrMatrixPlot():
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corrMatrix, annot = True, ax = ax, cmap="coolwarm")
    plt.show()


if __name__ == "__main__":
    main()



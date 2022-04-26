# Author: Shane O'Gorman
# Analysis of Iris Data set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os 

def main():
    ##### Importing data into dataframe #####
    collumnNames = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"] # List of the collumn names to be added to dataframe
    df = pd.read_csv('iris.txt', names = collumnNames) # Reads in dataset as df and adds the collumn names to top

    ##### Function calls 1 #####
    #os.mkdir("Plots") # Creates folder "Plots" - use once and comment out
    #summaryFile(df)
    #plotBoxPlots(df)
    #plotHistograms(df)
    #pairPlots(df)
    #plotScatterplots(df)
    #corrMatrixPlot(df)

    ##### Use df2 for machine learning work setting species to numerical values #####
    df2 = df
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df2["Species"] = le.fit_transform(df2["Species"])

    ##### Splits data into X and Y #####
    X = df2.drop(["Species"], axis=1)
    Y = df2["Species"]

    ##### Splits data into training and testing #####
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

    ##### Function calls 2 #####

    #logRegAccuracy(x_train, x_test, y_train, y_test)

    ##### Function calls 3 #####

    #max_k = KNNAccuracy(x_train, x_test, y_train, y_test)
    #KNNmodel(X, Y, max_k) # You will need to run KNNAccuracy with this function
  
##### Writes to summary file #####

def summaryFile(df):

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

##### Opens text file named summary.txt to write to as f #####
    with open("Summary.txt", "w") as f:
        
        ##### Writes to Summary.txt using variable above and converting them to strings using str #####
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

##### Plots Boxplots #####

def plotBoxPlots(df):
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.boxplot(data=df, x="Species", y="SepalLengthCm", ax=axs[0, 0], palette="Set2")
    sns.boxplot(data=df, x="Species", y="SepalWidthCm", ax=axs[0, 1], palette="Set2")
    sns.boxplot(data=df, x="Species", y="PetalLengthCm", ax=axs[1, 0], palette="Set2")
    sns.boxplot(data=df, x="Species", y="SepalWidthCm", ax=axs[1, 1], palette="Set2")
    
    plt.savefig("Plots/boxplots.png")
    #plt.show()

##### Plots Histograms #####

def plotHistograms(df):
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    sns.histplot(data=df, x="SepalLengthCm", kde=True, ax=axs[0, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="SepalWidthCm", kde=True, ax=axs[0, 1], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalLengthCm", kde=True, ax=axs[1, 0], hue="Species", palette="Set2")
    sns.histplot(data=df, x="PetalWidthCm", kde=True, ax=axs[1, 1], hue="Species", palette="Set2")
    
    plt.savefig("Plots/histograms.png")
    #plt.show()

##### Plots Pairplots #####

def pairPlots(df):
    sns.pairplot(df, kind="scatter", hue="Species", markers=["o", "s", "D"], palette="Set2")
    
    plt.savefig("Plots/pairplots.png")
    #plt.show()

##### Plots Scatterplots #####

def plotScatterplots(df):
    sns.set(style="darkgrid")
    sns.lmplot(data=df, x="PetalWidthCm", y="PetalLengthCm", fit_reg=False, hue="Species", markers=["o", "s", "D"], palette="Set2")
    
    plt.savefig("Plots/Petal-WidthxLength.png")
    #plt.show()

##### Plots Correlation Matrix #####

def corrMatrixPlot(df):
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corrMatrix, annot = True, ax = ax, cmap="coolwarm")
    
    plt.savefig("Plots/CorrelationMatrix.png")
    #plt.show()

##### Builds Logistic Regression Model, trains data on training data and returns model accuracy based on test data #####

def logRegAccuracy(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred1 = logreg.predict(x_test)

    from sklearn import metrics
    accuracy = (metrics.accuracy_score(y_test, y_pred1) * 100)
    print("Logistic Regression Model Accuracy: ", accuracy, "%" , sep="")

##### Finds the most accurate k to use for KNN model #####

def KNNAccuracy(x_train, x_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    scores = []
    k_range = []
    k_range.extend(range(1,26))

    from sklearn import metrics
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred2 = knn.predict(x_test)
        scores.append(metrics.accuracy_score(y_test, y_pred2))

    sns.set(style="darkgrid")
    plt.plot(k_range, scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Accurancy Scores")
    plt.show()

    k_range=[str(x) for x in k_range]
    zip_iterator = zip(k_range, scores)
    scoresDict = dict(zip_iterator)

    max_k = max(scoresDict, key=scoresDict.get)
    accuracy=scoresDict[max_k]*100
    print("KNN Model Accuracy where k=", max_k, ": ", accuracy, "%", sep="")
    return max_k

##### Builds KNN model where k has the highest accuracy in KNNAccuracy(). #####
##### Asks user to input Iris measurements and returns prediction #####

def KNNmodel(X, Y, max_k):
    from sklearn.neighbors import KNeighborsClassifier
    k=int(max_k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, Y)

    print("Please input the measurements of the Iris:")
    sl = int(input("Sepal Length in cm: "))
    sw = int(input("Sepal Width in cm: "))
    pl = int(input("Petal Length in cm: "))
    pw = int(input("Petal Width in cm: "))
    X_new = [sl, sw, pl, pw]

    prediction=knn.predict([X_new])

    if prediction[0]==0:
        print("Based on your inputted measurements, this is a Setosa.")
    elif prediction[0]==1:
        print("Based on your inputted measurements, this is a Versicolor.")
    else:
        print("Based on your inputted measurements, this is a Virginica.")

if __name__ == "__main__":
    main()



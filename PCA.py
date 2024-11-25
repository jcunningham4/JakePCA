import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

def meanvalue(df):
    return df.mean()


def covariance(x,y):
    xmean = meanvalue(x)
    ymean = meanvalue(y)
    for i in range(0,len(x)):
        x[i] = x[i] - xmean
    for i in range(0,len(y)):
        y[i] = y[i] - ymean
    return sum((x*y))/(len(x))


#def std(x):
#    return np.sqrt(covariance(x,x))


def standardise(x):
    return ((x-x.mean())/x.std())


def covmatrix(x):
    matrix = np.zeros((4,4))
    for i in range(0,4):
        for d in range(0,4):
            matrix[i,d] = covariance(x[i],x[d])
    return(matrix)

def eigenfunction(eigenvector):
    magnitudes = []
    for i in range(len(eigenvector)):
        magnitude_i = 0
        for j in range(len(eigenvector[i])):
            coefficient_ij = eigenvector[i,j]
            magnitude_i = magnitude_i +(coefficient_ij**2)
        magnitudes.append(np.sqrt(magnitude_i))
    return magnitudes
    
def sort_eigens(eigenvalues, eigenvectors):
    df_eigen = pd.DataFrame(eigenvectors)
    df_eigen['Eigenvalues'] = eigenvalues
    df_eigen.sort_values("Eigenvalues", inplace=True, ascending=False)
    sorted_eigenvalues = np.array(df_eigen['Eigenvalues'])
    sorted_eigenvectors = np.array(df_eigen.drop(columns="Eigenvalues"))

    return sorted_eigenvalues, sorted_eigenvectors

def reorient_data(df,eigenvectors):
    # turns the dataframe into a numpy array to enable matrix multiplication
    numpy_data = np.array(df)

    # mutiplies the data by the eigenvectors to get the data in terms of pca features
    pca_features = np.dot(numpy_data, eigenvectors)

    # turns the new array back into a dataframe for plotting
    pca_df = pd.DataFrame(pca_features)

    return pca_df

x = df["sepal length (cm)"]
y = df["sepal width (cm)"]
z = df["petal length (cm)"]
a = df["petal width (cm)"]

fig, ax = plt.subplots()
#fig.suptitle("sepal length vs sepal width", size=14)
#ax.plot(x,y, "k.")
#ax.set_xlabel("sepal length (cm)")
#ax.set_ylabel("sepal width (cm)")

matrix = covmatrix((x,y,z,a))
stdmatrix = covmatrix((standardise(x),standardise(y),standardise(z),standardise(a)))

eigenvalue,eigenvector = eig(matrix)
stdeigenvalue, stdeigenvector = eig(stdmatrix)

#eigenfunction(eigenvector)
#eigenfunction(stdeigenvector)
sorted_eigenvalue, sorted_eigenvector = sort_eigens(eigenvalue,eigenvector)
df = reorient_data(df,eigenvector)
print(df)

x = df[0]
y = df[1]
z = df[2]
a = df[3]


fig.suptitle("PC2 vs PC3", size=14)
ax.plot(a,z, "k.")
ax.set_xlabel("PC2")
ax.set_ylabel("PC3")
plt.show()
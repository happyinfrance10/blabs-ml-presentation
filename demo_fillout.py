import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing

#a. KNN
from sklearn.neighbors import KNeighborsClassifier

def knn(X, y):
    #TODO

#b. Decision tree
from sklearn.tree import DecisionTreeClassifier

def dtc(X, y):
    #TODO

#c. Random forest
from sklearn.ensemble import RandomForestClassifier

def rfc(X, y):
    #TODO

#d. linear regression
from sklearn.linear_model import LinearRegression

def lin(X, y):
    #TODO

#e. logistic regression
from sklearn.linear_model import LogisticRegression

def log(X, y):
    #TODO

#f. K means
from sklearn.cluster import KMeans

def kme(X):
    #TODO

def evaluate_cat(X, y, modtype="knn"):
    #TODO

def evaluate_reg(X, y, modtype="lin"):
    #TODO

def main():
    iris = load_iris(return_X_y=True)
    knn_e = evaluate_cat(iris[0], iris[1], modtype="knn")
    print('KNN accuracy: {}'.format(knn_e))
    dtc_e = evaluate_cat(iris[0], iris[1], modtype="dtc")
    print('DTC accuracy: {}'.format(dtc_e))
    rfc_e = evaluate_cat(iris[0], iris[1], modtype="rfc")
    print('RFC accuracy: {}'.format(rfc_e))
    log_e = evaluate_cat(iris[0], iris[1], modtype="log")
    print('LOG accuracy: {}'.format(log_e))

    calh = fetch_california_housing(return_X_y=True)
    lin_e = evaluate_reg(calh[0], calh[1], modtype="lin")
    print('LIN mean squared error: {}'.format(lin_e))

    kme_data = np.array([[2.1,4],[5.2,6.8],[7.4,9.1],[1.3,5.7],[4.6,5.9],[-0.9,6.1],[2.5,6],[6.7,2.4],
                        [2.8,3.4],[7.2,6.9],[7.7,8.8],[2.3,10.7],[1.6,-5.9],[2.9,-3.1],[4.5,4.6],
                        [12.2,4.9],[5.1,1.3],[-7.4,3.2],[-1.3,0.7],[4.7,8.9],[3.9,4.1],[0.5,7.1],
                        [4.0,2.3],[4.5,6.7],[3.2,2.7]])
    mod_kme = kme(kme_data)
    print('Cluster labels (KME): {}'.format(repr(mod_kme.labels_)))
    print('Cluster centers (KME): {}'.format(repr(mod_kme.cluster_centers_)))

if __name__=="__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.datasets import load_iris
iris = load_iris()


x_train, x_test, y_train, y_test = train_test_split(iris['data'],iris['target'],random_state=0)
knn = knc(n_neighbors=1)
knn.fit(x_train, y_train)
print('Enter the values')
a=list(map(float,input().split()))
x_new = np.array([a])
val = int(knn.predict(x_new))
if(val == 0):
    print('Setosa')
elif(val == 1):
    print('Virginica')
elif(val == 2):
    print('Versicolor')
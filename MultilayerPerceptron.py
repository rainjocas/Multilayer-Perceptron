"""
Author: Rain Jocas
Date: 2/1/2025
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

def print_scores():
    np.set_printoptions( suppress=True,
    precision =2)
    print(clf.predict_proba(X_test[ 0:3, :]))
    print(clf.predict(X_test[ 0:3, :]))
    print(y_test[0:3])
    print(clf.score(X_test , y_test))

def plot_decision_boundaries(X, y, model_class, **model_params):
    model = model_class(**model_params, random_state=10)
    model.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .01 * np.mean([x_max - x_min, y_max - y_min])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()

#make dataset
X, y = make_blobs( n_samples=400, centers=4, cluster_std =2, random_state =10)

#train data
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size =0.4, random_state =2)

#visualize data
plt.scatter(X_train[: , 0], X_train[: , 1], c =y_train)
plt.title( "Training Data for make_blobs")
plt.show()

#make dataset
X, y = make_circles(n_samples=100, shuffle=True, noise=0.02, random_state=None, factor=0.8)

#train data
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size =0.4, random_state =2)

#visualize data
plt.scatter(X_train[: , 0], X_train[: , 1], c =y_train)
plt.title( "Training Data for make_circles" )
plt.show()


clf = MLPClassifier( hidden_layer_sizes =5, random_state =10)
clf.fit(X_train , y_train) # This command trains the MLP on the training data


print_scores()

plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes =(50, 20, 30))
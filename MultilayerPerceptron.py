"""
Author: Rain Jocas, Temi Agunloye
Date: 2/1/2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import time

def print_scores():
    np.set_printoptions( suppress=True,
    precision =2)
    print(clf.predict_proba(X_test[ 0:3, :]))
    print(clf.predict(X_test[ 0:3, :]))
    print(y_test[0:3])
    print(clf.score(X_test , y_test))

def plot_decision_boundaries(X, y, model_class, plotTitle, **model_params):
    """
    Plots decision boundries as given by MLP predictions

    Author: (We took this from the slides and made some minor tweaks)
    """
    model = model_class(**model_params, random_state=10)
    model.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .01 * np.mean([x_max - x_min, y_max - y_min])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(plotTitle)
    plt.show()


def compare_optimization(X, y, X_train , y_train):
    """ Temi Agunloye
    
    NOTE: Add Documentation
    """

    plot_decision_boundaries(X_train, y_train, MLPClassifier, hidden_layer_sizes =(50, 20, 30))

def network_architecture_test(X, y, X_train, X_test, y_train, y_test, layerSize, activationType, plotTitle):
    """
    Runs MLP given required data and network architecture parameters. Returns various metrics

    Author: Rain Jocas
    """
    startTime = time.time()
    clf = MLPClassifier(hidden_layer_sizes = layerSize, activation = activationType, random_state =10)
    clf.fit(X_train , y_train) # This command trains the MLP on the training data
    score = clf.score(X_test, y_test)
    bestLoss = clf.best_loss_ #The minimum loss reached by the solver throughout fitting
    endTime = time.time()
    runTime = endTime - startTime
    plot_decision_boundaries(X_train, y_train, MLPClassifier, plotTitle, hidden_layer_sizes = layerSize)
    return (score, bestLoss, runTime)

def store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime):
    """
    A quick helper function that stores the results of a test
    
    Author: Rain Jocas
    """
    scores.append(score)
    bestLosses.append(bestLoss)
    runTimes.append(runTime)

def n_layers_test(X, y, X_train, X_test, y_train, y_test):
    """
    Tests MLP on different numbers of layers and returns a dataframe various metrics

    Author: Rain Jocas
    """
    scores = []
    bestLosses = []
    runTimes = []
    
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test, (10),
                                                         'identity', "1 Layer")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test, (10, 10),
                                                         'identity', "2 Layers")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test, (10, 10, 10),
                                                         'identity', "3 Layers")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test, (10, 10, 10, 10),
                                                         'identity', "4 Layers")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test, (10, 10, 10, 10, 10),
                                                         'identity', "5 Layers")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)

    #turn into dataframe
    results = {'Number of Layers': [1, 2, 3, 4, 5], 'Scores': scores, 'Best Loss': bestLosses, 'Run Times': runTimes}
    results = pd.DataFrame(results)
    return results

def activation_test(X, y, X_train, X_test, y_train, y_test):
    """
    Tests MLP with different activation types and returns a dataframe various metrics

    Author: Rain Jocas
    """
    scores = []
    bestLosses = []
    runTimes = []

    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test,
                                                         (10, 10, 10), 'identity', "Activation Type- Identity")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test,
                                                         (10, 10, 10), 'logistic', "Activation Type- Logistic")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test,
                                                         (10, 10, 10), 'tanh', "Activation Type- Tanh")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)
    score, bestLoss, runTime = network_architecture_test(X, y, X_train, X_test, y_train, y_test,
                                                         (10, 10, 10), 'relu', "Activation Type- Relu")
    store_scores(scores, bestLosses, runTimes, score, bestLoss, runTime)

    #turn into dataframe
    results = {'Activation type': ['Identity', 'Logistic', 'Tanh', 'Relu'],
               'Scores': scores, 'Best Loss': bestLosses, 'Run Times': runTimes}
    results = pd.DataFrame(results)
    return results


def compare_network_architecture(X, y, X_train, X_test, y_train, y_test):
    """
    Runs layer and activation type testing and prints results

    Author: Rain Jocas
    """
    nLayers = n_layers_test(X, y, X_train, X_test, y_train, y_test)
    activations = activation_test(X, y, X_train, X_test, y_train, y_test)
    print("\n", nLayers)
    print("\n", activations, "\n")
    

def run_all():
    """
    Runs all network architecture and optimization tests for the make_blobs and make_circles datasets

    Authors: Rain Jocas, Temi Agunloye 
    """
    #make dataset
    X, y = make_blobs( n_samples=400, centers=4, cluster_std =2, random_state =10)

    #train data
    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size =0.4, random_state =2)

    #visualize data
    plt.scatter(X_train[: , 0], X_train[: , 1], c =y_train)
    plt.title( "Training Data for make_blobs")
    plt.show()

    #Run Network Architecture tests
    compare_network_architecture(X, y, X_train, X_test, y_train, y_test)

    #make dataset
    X, y = make_circles(n_samples=400, shuffle=True, noise=0.08, random_state=None, factor=0.8)

    #train data
    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size =0.4, random_state =2)

    #visualize data
    plt.scatter(X_train[: , 0], X_train[: , 1], c =y_train)
    plt.title( "Training Data for make_circles" )
    plt.show()

    #Run Network Architecture tests
    compare_network_architecture(X, y, X_train, X_test, y_train, y_test)

run_all()
from sklearn import metrics

import met
import numpy as np


# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 23:16:03 2018

@author: Raneem
"""


def purity_measure(y_true, y_pred):
    """    
    The MIT License (MIT)
    Copyright (c) 2017 David Mugisha

    Purity score

    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.abs

    Parameters
    ----------
        y_true: np.ndarray: n*1 matrix Ground truth labels
        y_pred: np.ndarray: n*1 matrix Predicted clusters
    
    Returns
    -------
        Purity score: float
    
    References
    ----------
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.min(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmin(hist)
        y_labeled_voted[y_pred==cluster] = winner
    
    return metrics.accuracy_score(y_true, y_labeled_voted)



def TWCV(points, labelsPred): 
    """    
    Total Within Cluster Variance measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global k
    sumAllFeatures = sum(sum(np.power(points,2)))
    sumAllPairPointsCluster = 0
    for clusterId in range(k):
        indices = np.where(labelsPred == clusterId)[0]
        pointsInCluster = points[np.array(indices)]
        sumPairPointsCluster = sum(pointsInCluster)
        sumPairPointsCluster = np.power(sumPairPointsCluster,2)
        sumPairPointsCluster = sum(sumPairPointsCluster)
        sumPairPointsCluster = sumPairPointsCluster/len(pointsInCluster)
        
        sumAllPairPointsCluster += sumPairPointsCluster
        fitness = (sumAllFeatures - sumAllPairPointsCluster)
    return fitness


def SC(points, labelsPred):
    """    
    Silhouette Coefficient measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global fitnessFunc
    fitnessFunc = "Silhouette"    
    silhouette = metrics.silhouette_score(points, labelsPred, metric='euclidean')
    
    #silhouette = (silhouette - (-1)) / (1 - (-1))
    silhouette = (silhouette + 1) / 2
    fitness = 1 - silhouette
    return fitness

def SSE(assignersDistances):
    """    
    Sum of Squared Error measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global fitnessFunc
    fitnessFunc = "SSE"
    fitness = np.sum(np.square(assignersDistances[assignersDistances < float("inf")]))
    return fitness


def DB(points, labelsPred):
    """    
    Davies–Bouldin measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global fitnessFunc
    fitnessFunc = "DB"
    fitness = metrics.davies_bouldin_score(points, labelsPred)
    return fitness

def CHI(points, labelsPred):
    """    
    Calinski and Harabasz Index measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global fitnessFunc
    fitnessFunc = "CH"
    ch = metrics.calinski_harabaz_score(points, labelsPred)
    fitness = 1 / ch
    return fitness


def DI(points, labelsPred):
    """    
    Dunn Index measure
    
    Parameters
    ---------- 
    point : ndarray
        The point that we need to find its Nearest
    labelPred : list
        A list of predicted labels for the points for each chroomosome
            
    Returns
    -------
    float
        fitness: The fitness value
    """
    global fitnessFunc
    fitnessFunc = "Dunn"
    dunn = met.dunn_fast(points, labelsPred)
    if(dunn < 0):
        dunn = 0
    fitness = 1 - dunn
    return fitness
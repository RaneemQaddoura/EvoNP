""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                                                                        """
"""                                   EvoNP                                """
"""  Evolutionary-based algorithm with Nearest point Clustering Technique  """
"""                             @author: Raneem                            """
"""                                version 1.0                             """
"""                                                                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(__doc__)

from NP import NP
from scipy import spatial

import random
import fitness as fit
import EAoperators
import numpy as np
import sys

def run(points,nPoints, k, nChromosomes, nGenerations, crossoverProbability, mutationProbability):
    """
    The main method of EvoNP

    Parameters
    ----------    
    points : list
        The list of points. A two dimensional array where each row represents 
        a point containing the features values for the point    
    nPoints:
        Number of points (instances) in the dataset        
    k : int
        Number of clusters    
    nChromosomes: int
        Number of chrmosome in a population
    nGenerations: int
        Number of generations        
    crossoverProbability : float
        The probability of crossover
    mutationProbability : float
        The probability of mutation
    
    
    Returns
    -------
    int
        bestChromosomeInAllGenerations: The chromosome with the minimum fitness in all the generations
    list
        bestLabelsPredInAllGenerations: The predicted label of the chromosome with the minimum fitness in all the generations
    float
        bestFitnessInAllGenerations: The minimum fitness in all the generations
    list
        sumBestFitness: A matrix of all the values of the fitness for each generation
    """
    
    #pointsNearestIndices is a matrix of nearest points for each previously elected point
    #pointsNearestDistances is a matrix of the distances between the nearest points and the Elected for each previously elected point
    pointsNearestIndices =  [[n] for n in range(nPoints)]
    pointsNearestDistances =  [[0] for n in range(nPoints)]
        
    #sortedDistancesTree is a dimensional tree of all the points sorted in a way where the nearest point to another point is easily retrieved
    sortedDistancesTree = spatial.cKDTree(points)          

    #stores all the best fitness values for every generation    
    sumBestFitness = [0]*nGenerations
    
    # Creating the initial population.
    population = [random.sample(range(0, nPoints), k) for i in range(nChromosomes)]
    
    #initializing variables
    bestChromosomeInAllGenerations = None
    bestLabelsPredInAllGenerations = None
    bestFitnessInAllGenerations = 9999999999
    
    #loop through generations    
    for generation in range(nGenerations - 1):
        
        # Applying NP search technique and measering the fitness of each chromosome in the population.        
        population,fitness, labelsPred, bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations = generationRun(points, population, k, nChromosomes, sortedDistancesTree, 
                  pointsNearestIndices, pointsNearestDistances,
                  bestChromosomeInAllGenerations, bestFitnessInAllGenerations,
                  bestLabelsPredInAllGenerations)
        
        # The best result in the current iteration.
        bestFitnessInGeneration = np.min(fitness)              
        sumBestFitness[generation] = sumBestFitness[generation] + bestFitnessInGeneration
        
        #Apply evolutionary operators to Updated chromosomes
        population = EAoperators.runOperators(population, fitness,
                                      crossoverProbability, mutationProbability, 
                                      nChromosomes, nPoints) 
    
    # Getting the best solution after iterating finishing all generations.
    population,fitness, labelsPred,bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations = generationRun(points, population, k, nChromosomes, sortedDistancesTree, 
                  pointsNearestIndices, pointsNearestDistances,
                  bestChromosomeInAllGenerations, bestFitnessInAllGenerations,
                  bestLabelsPredInAllGenerations)
        
    # The best result for all generations
    bestFitness = np.min(fitness)
    bestChromosome = np.where(fitness == bestFitness)[0][0]
    bestLabelsPredInAllGenerations = labelsPred    
    bestFitnessInAllGenerations = bestFitness
    bestChromosomeInAllGenerations = bestChromosome
    sumBestFitness[generation] = sumBestFitness[generation] + bestFitness

    generation = generation + 1
    
    return bestChromosomeInAllGenerations, bestLabelsPredInAllGenerations, bestFitnessInAllGenerations, sumBestFitness


def generationRun(points, population, k, nChromosomes, sortedDistancesTree, 
                  pointsNearestIndices, pointsNearestDistances,
                  bestChromosomeInAllGenerations, bestFitnessInAllGenerations,
                  bestLabelsPredInAllGenerations):
    """
    The main method of EvoNP

    Parameters
    ----------    
    points : list
        The list of points. A two dimensional array where each row represents 
        a point containing the features values for the point    
    population : list
        The list of chromosomes
    k : int
        Number of clusters    
    nChromosomes: int
        Number of chrmosome in a population
    sortedDistancesTree : cKDTree
        A tree of all the points sorted in a way where the nearest point to another point is easily retrieved
    pointsNearestIndices : list
        A matrix of nearest points for each previously elected point
    pointsNearestDistances : list
        A matrix of the distances between the nearest points and the Elected for each previously elected point
    bestChromosomeInAllGenerations : int
        The chromosome with the minimum fitness in all the generations
    bestFitnessInAllGenerations: float
        The minimum fitness in all the generations
    bestLabelsPredInAllGenerations: list
        The predicted label of the chromosome with the minimum fitness in all the generations
    
    
    Returns
    -------
    list
        population : The list of chromosomes
    list
        fitness : The list of fitness value for each chromosome 
    list
        labelsPred : The list of predicted labels
    int
        bestChromosomeInAllGenerations : The chromosome with the minimum fitness in all the generations
    float
        bestFitnessInAllGenerations : The minimum fitness in all the generations
    list
        bestLabelsPredInAllGenerations : The predicted label of the chromosome with the minimum fitness in all the generations
    """
    
    #initializing variables
    fitness = [None] * nChromosomes
    labelsPred = [None] * nChromosomes
    
    #Loop through chromosomes in population
    for chromosomeId in range(nChromosomes):
        
        #Get Initial and elected points from chromosome
        initialRandomPoints = population[chromosomeId].copy()
        initialPointsIndices = initialRandomPoints[:k].copy()
        randomPointsIndices = initialRandomPoints[k:].copy()
        
        if len(np.unique(initialPointsIndices)) < k:
            #Duplicates in initial points. We consider it as bad chromosome 
            labelsPred[chromosomeId] = None
            fitness[chromosomeId] = 9999999999
        else:
            
            # Applying NP search technique on chromosome in the population. 
            labelsPredList, chromosome, pointsNearestIndices, pointsNearestDistances, assignersDistances = NP(points, 
                                                                                          sortedDistancesTree, k, 
                                                                                          initialPointsIndices, randomPointsIndices, 
                                                                                          pointsNearestIndices, pointsNearestDistances)
            labelsPred[chromosomeId] = labelsPredList.copy()
            #Measering the fitness of each chromosome in the population.        
            fitness[chromosomeId] = fit.SC(points,labelsPredList)
            
            population[chromosomeId] = chromosome.copy()
            
    #Elitism operation
    bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations = EAoperators.elitism(population, fitness, labelsPred, bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations)
    
    return population,fitness, labelsPred, bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations

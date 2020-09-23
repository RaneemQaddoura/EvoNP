"""
Created on Wed Nov  7 20:18:05 2018

@author: Raneem
"""
import numpy as np
import random

def runOperators(population, fitness, 
                  crossoverProbability, mutationProbability, 
                  nChromosomes, nPoints):
    """    
    This is the main method where the evolutionary operators are called
    
    Parameters
    ----------    
    population : list
        The list of chromosomes
    fitness : list
        The list of fitness values for each chromosome
    crossoverProbability : float
        The probability of crossover
    mutationProbability : float
        The probability of mutation
    nChromosomes: int
        Number of chrmosome in a population
    nPoints:
        Number of points (instances) in the dataset
    
    Returns
    -------
    list
        newPopulation: the new generated population after applying the genetic operations
    """
    
    #initialize a new population
    newPopulation = [None] * nChromosomes    
    
    #Create pairs of parents. The number of pairs equals the number of chromosomes divided by 2
    for i  in range(0, nChromosomes, 2):
        #pair of parents selection
        parent1, parent2 = pairSelection(population, fitness, nChromosomes)
        
        #crossover
        crossoverLength = min(len(parent1), len(parent2))
        parentsCrossoverProbability = random.uniform(0.0, 1.0)
        if parentsCrossoverProbability < crossoverProbability:
            offspring1, offspring2 = crossover(crossoverLength, parent1, parent2)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
        
        #Mutation   
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < mutationProbability:
            mutation(offspring1, len(offspring1), nPoints)
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < mutationProbability:
            mutation(offspring2, len(offspring2), nPoints)
        
        #Add offsprings to population
        newPopulation[i] = offspring1.copy()
        newPopulation[i + 1] = offspring2.copy()     
    
    return newPopulation

def elitism(population, fitness, labelsPred, bestChromosomeInAllGenerations, 
            bestFitnessInAllGenerations, bestLabelsPredInAllGenerations):
    """    
    This method performs the elitism operator
    
    Parameters
    ----------    
    population : list
        The list of chromosomes
    fitness : list
        The list of fitness values for each chromosome
    labelPred : list
        A list of predicted labels for the points for each chroomosome
    bestChromosomeInAllGenerations : list
        A chromosome of the previous generation having the best fitness value          
    bestFitnessInAllGenerations : float
        The best fitness value of the previous generation        
    bestLabelsPredInAllGenerations :
        A list of predicted labels for the previous generation having the best fitness value          

    Returns
    -------
    list
        population : The updated population after applying the elitism
    list
        fitness : The updated list of fitness values for each chromosome after applying the elitism
    list
        labelsPred : The updated list of predicted labels for the points for each chroomosome after applying the elitism
    list
        bestChromosomeInAllGenerations : A chromosome of the current generation having the best fitness value          
    float
        bestFitnessInAllGenerations : The best fitness value of the current generation        
    list
        bestLabelsPredInAllGenerations : A list of predicted labels for the current generation having the best fitness value          
    """
    
    # get the worst chromosome
    worstFitnessId = selectWorstChromosome(fitness)
    
    #replace worst cromosome with best one from previous generation if its fitness is less than the other
    if fitness[worstFitnessId] > bestFitnessInAllGenerations:
        population[worstFitnessId] = bestChromosomeInAllGenerations.copy()
        fitness[worstFitnessId] = bestFitnessInAllGenerations
        labelsPred[worstFitnessId] = bestLabelsPredInAllGenerations.copy()
    
    #update best chromosome
    bestFitnessId = selectBestChromosome(fitness)
    bestChromosomeInAllGenerations = population[bestFitnessId].copy()
    bestFitnessInAllGenerations = fitness[bestFitnessId].copy()
    bestLabelsPredInAllGenerations = labelsPred[bestFitnessId].copy()
    
    return bestChromosomeInAllGenerations, bestFitnessInAllGenerations, bestLabelsPredInAllGenerations


def selectWorstChromosome(fitness):
    """    
    It is used to get the worst chromosome in a population based n the fitness value
    
    Parameters
    ---------- 
    fitness : list
        The list of fitness values for each chromosome
        
    Returns
    -------
    int
        maxFitnessId: The chromosome id of the worst fitness value
    """
    
    maxFitnessId = np.where(fitness == np.max(fitness))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId

def selectBestChromosome(fitness):
    """    
    It is used to get the best chromosome in a population based n the fitness value
    
    Parameters
    ---------- 
    fitness : list
        The list of fitness values for each chromosome
        
    Returns
    -------
    int
        maxFitnessId: The chromosome id of the best fitness value
    """
    minFitnessId = np.where(fitness == np.min(fitness))
    minFitnessId = minFitnessId[0][0]
    return minFitnessId

def pairSelection(population, fitness, nChromosomes):    
    """    
    This is used to select one pair of parents using roulette Wheel Selection mechanism
    
    Parameters
    ---------- 
    population : list
        The list of chromosomes
    fitness : list
        The list of fitness values for each chromosome
    nChromosomes: int
        Number of chrmosome in a population
          
    Returns
    -------
    list
        parent1: The first parent chromosome of the pair
    list
        parent2: The second parent chromosome of the pair
    """
    sumFitness = sum(fit for fit in fitness)
    parent1 = population[rouletteWheelSelectionId(fitness, sumFitness, nChromosomes)]
    parent2 = population[rouletteWheelSelectionId(fitness, sumFitness, nChromosomes)]
    
    return parent1, parent2
    
def rouletteWheelSelectionId(fitness, sumFitness, nChromosomes): 
    """    
    A roulette Wheel Selection mechanism for selecting a chromosome
    
    Parameters
    ---------- 
    fitness : list
        The list of fitness values for each chromosome
    sumFitness : float
        The summation of all the fitness values for all chromosomes in a generation
    nChromosomes: int
        Number of chrmosome in a population
          
    Returns
    -------
    id
        chromosomeId: The id of the chromosome selected
    """
    pick    = random.uniform(0, sumFitness)
    current = 0
    for chromosomeId in range(nChromosomes):
        current += fitness[chromosomeId]
        if current > pick:
            return chromosomeId

def crossover(chromosomeLength, parent1, parent2):
    """    
    The crossover operator
    
    Parameters
    ---------- 
    parent1 : list
        The first parent chromosome of the pair
    parent2 : list
        The second parent chromosome of the pair
    chromosomeLength: int
        The maximum index of the crossover
          
    Returns
    -------
    list
        offspring1: The first updated parent chromosome of the pair
    list
        offspring2: The second updated parent chromosome of the pair
    """
    
    # The point at which crossover takes place between two parents. 
    crossover_point = random.randint(0, chromosomeLength - 1)

    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
    offspring1 = parent1[0:crossover_point] +  parent2[crossover_point:]
    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
    offspring2 = parent2[0:crossover_point] +  parent1[crossover_point:]
      
    return offspring1, offspring2


def mutation(offspring, chromosomeLength, nPoints):
    """    
    The mutation operator
    
    Parameters
    ---------- 
    offspring : list
        A generated chromosome after the crossover
    chromosomeLength: int
        The maximum index of the crossover
    nPoints:
        Number of points (instances) in the dataset
         
    Returns
    -------
    list
        offspring: The updated offspring chromosome
    """
    mutationIndex = random.randint(0, chromosomeLength - 1)
    mutationValue = random.randint(0, nPoints - 1)
    offspring[mutationIndex] = mutationValue
    #return offspring

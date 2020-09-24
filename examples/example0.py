import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/evonp')))

import EvoNP
from sklearn import metrics
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#initializing variables
directory = "../datasets/" # the directory where the dataset is stored
nChromosomes = 20 
nGenerations = 50 
crossoverProbability = 0.8
mutationProbability = 0.001

filename = "aggregation.csv"

# Read the dataset file and generate the points list and true values
data = np.genfromtxt(directory + filename, delimiter=',')

nPoints, nValues = data.shape #Number of points and Number of values for each point
nValues = nValues - 1 #Dimension value
k = len(np.unique(data[:,-1]))#k: Number of clusters
points = data[:,:-1] #list of points
labelsTrue = data[:,-1] #List of actual cluster of each points (last field)
     
popSize = (nChromosomes,k) # The population will have nChromosomes chromosome where each chromosome has chromosomeLength genes.

bestChromosomeInAllGenerations, bestLabelsPredInAllGenerations, bestFitnessInAllGenerations, allBestFitness = EvoNP.run(points, nPoints, k, nChromosomes, nGenerations, crossoverProbability, mutationProbability)

print("HS: " + str(float("%0.2f"%metrics.homogeneity_score(labelsTrue,bestLabelsPredInAllGenerations[bestChromosomeInAllGenerations]))))
print("CS: " + str(float("%0.2f"%metrics.completeness_score(labelsTrue,bestLabelsPredInAllGenerations[bestChromosomeInAllGenerations]))))
print("VM: " + str(float("%0.2f"%metrics.v_measure_score(labelsTrue,bestLabelsPredInAllGenerations[bestChromosomeInAllGenerations]))))
print("AMI: " + str(float("%0.2f"%metrics.adjusted_mutual_info_score(labelsTrue,bestLabelsPredInAllGenerations[bestChromosomeInAllGenerations]))))
print("ARI: " + str(float("%0.2f"%metrics.adjusted_rand_score(labelsTrue,bestLabelsPredInAllGenerations[bestChromosomeInAllGenerations]))))

# plot fitness progression
allGenerations = [x+1 for x in range(nGenerations)]
plt.plot(allGenerations, allBestFitness)
plt.title(filename[:-4])
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.grid()
#plt.savefig("test.png")
plt.show()
plt.clf()
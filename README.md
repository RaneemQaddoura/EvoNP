# Evolutionary Algorithm with Nearest Point (EvoNP)
An efficient evolutionary algorithm with Nearest Point is a clustering algorithm which aims at grouping similar data points to the same cluster and dissimilar data points to different clusters. It is based on the evolution behavior of genetic algorithm and the Nearest Neighbor Search (NNS) technique. The algorithm starts by reading the data set and generating the initial population. The initial population is then passed to the Nearest Point (NP) clustering technique where the Election, Selection, and Assignment operators generating an updated population which is then evaluated using a specific fitness function. The population is then passed to the evolutionary operators where the selection, crossover, mutation, and elitism operators are performed generating an evolved population. The evolved population is then considered as a new population for the next round of evolving using the NP clustering technique and evolutionary operators until a predefined number of generations is reached. The best chromosome from the last generation is considered as the final best solution. 

## Requirements
- sklearn: 0.23
- NumPy: 1.19.0
- SciPy: 1.5.2
- Matplotlib: 3.3.0rc1
- Pandas: 1.1.2
- treelib: 1.5.5

## Setup virtual enviroment
If you would like to work in an isolated environment which is strongly recommended so that you can work on different projects without having conflicting library versions, install virtualenv by running the following pip command.
```
python3 -m pip install --user -U virtualenv
```

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy`, `SciPy`, `Matplotlib`, `Pandas`, and `treelib` for you.

- If you are installing EvoNP algorithm onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/RaneemQaddoura/EvoNP


## Quick User Guide

EvoNP contains the main file is the EvoNP.py, which represents the implementation of the algorithm. The test0.py is an example file for using the EvoNP algorithm as an interface to the algorithm. In the test0.py you can setup your experiment by selecting the datasets, number of chromosomes, number of generations, crossover probability, mutation probability, and number of runs. 

The following is a sample example to use the EvoNP algorithm.  

Change dataset names, number of chromosomes, number of generations, crossover probability, mutation probability, and number of runs variables as you want:  
```
dataset_List = ["VaryDensity.csv","flame.csv"]
nChromosomes = 20
nGenerations = 50
crossoverProbability = 0.8
mutationProbability = 0.001
runs = 30
```

Now your experiment is ready to run. Enjoy!

## EvoNP description page
http://evo-ml.com/evonp/

## Published Article

## Demo video

## Citing EvoNP


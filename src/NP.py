from treelib import Tree

import sys
import numpy as np
import random
  

def NP(points_, sortedDistancesTree_, k_, 
       initialPointsIndices_, randomPointsIndices_, 
       pointsNearestIndices_, pointsNearestDistances_):
    """    
    This is the implementation of the NP clustering technique
    
    Parameters
    ----------    
    points_ : list
        The list of points. A two dimensional array where each row represents 
        a point containing the features values for the point
    sortedDistancesTree_ : cKDTree
        A tree of all the points sorted in a way where the nearest point to another point is easily retrieved
    k : int
        Number of clusters    
    initialPointsIndices_ : list
        The indices of the initial points of the clusters which are retrieved from the first part of the chromosome
    randomPointsIndices_ : list
        The indices of the elected points of the clusters which are retrieved from the second part of the chromosome
    pointsNearestIndices_ : list
        A matrix of nearest points for each previously elected point
    pointsNearestDistances_ : list
        A matrix of the distances between the nearest points and the Elected for each previously elected point
    
    Returns
    -------
    list
        labelsPred: the predicted values of the data
    list
        chromosome: the updated chromosome generated after applying NP
    list
        pointsNearestIndices_ :  A matrix of the updated nearest points for each previously elected point
    list
        pointsNearestDistances_ : A matrix of the updated distances between the nearest points and the Elected for each previously elected point
    
    """
    global points, k, initialPointsIndices, randomPointsIndices    
    global sortedDistancesTree, pointsNearestIndices, pointsNearestDistances
    global labelsPred, assignersDistances
    
    #setting global variabless from algorithm parameters
    points = points_
    sortedDistancesTree = sortedDistancesTree_
    k = k_
    initialPointsIndices = initialPointsIndices_
    randomPointsIndices = randomPointsIndices_
    pointsNearestIndices = pointsNearestIndices_
    pointsNearestDistances = pointsNearestDistances_
    
    #initilizing the clusters
    init()
    
    #performing the calculations of the algorithm including the election, selection, and assignment operators 
    calculate()
    
    #combining the genes of the chromosome
    chromosome = initialPointsIndices +  randomPointsIndices
    
    return labelsPred, chromosome, pointsNearestIndices, pointsNearestDistances, assignersDistances

def init():
    """    
    Initializes the variables and data structures and creates the initial points for the clusters
    
    Parameters
    ----------    
    N/A
    
    Returns
    -------
    N/A
    """
    global nPoints, nValues, labelsPred, assignedPoints
    global remainingPoints, distanceVectorIndex
    global nAssignedPoints, assignmentsTree
    global nRemainingPoints, sortedDistancesTree, pointsNearestIndices, pointsNearestDistances
    global maxIndex, nElections, assignersIndices, assignersDistances
    
    #initialize variables and data structures
    nPoints = len(points) #Number of points 
    nValues = len(points[0]) #Dimension value. Number of features for each point
    assignmentsTree = Tree() #The tree containing the assignements btween points
    nAssignedPoints = 0 #Number of points that are already assigned by the algorithm to a certain cluster (initially equals 0)
    nRemainingPoints = nPoints #Number of points that are waiting to be assigned by the algorithm to a certain cluster (initially equals to the number of points)
    labelsPred = np.array([None] * nPoints) #List of predicted cluster value for each point (initially equals None for every point)
    assignedPoints = np.array([], dtype=int) #list containing the points that are already assigned by the algorithm to a certain cluster
    remainingPoints = np.array(range(nPoints), dtype=int) #list containing the points that are not yet assigned by the algorithm to a cluster
    nElections = 0 #Number of elections (initially equals 0 elections)
    assignersIndices = np.array([None] * nPoints) #The indices of the assigner for each point (initially equals None for every point indicating that no Assigner is yet specified for a point)
    assignersDistances = np.array([float("inf")] * nPoints) #The distances between each point and its assigner the assigner for each point (initially equals infinity for every point indicating that no Assigner is yet specified for a point)
    distanceVectorIndex = [1] * nPoints #Distance Vector Index initially equals 1 for every point. This is used to keep track of the number of nearest points selected for each elected point
    
    #Generate initial points to the clusters from the chromosome
    createInitialPoints()
  
def createInitialPoints():
    """
    Creates initial points for each cluster
    
    Parameters
    ----------    
    N/A
    
    Returns
    -------
    N/A
    """
    global k, nRemainingPoints, remainingPoints, assignmentsTree, initialPointsIndices
    
    #initialize the assignemnet tree containing the the initial points
    assignmentsTree.create_node("root","root")
    
    #assign each initial point to a cluster
    for clusterNo in range(k):
        initialPointIndex = initialPointsIndices[clusterNo]
        assignerIndex = -1
        distance = float("inf")
        addPointToCluster(clusterNo,initialPointIndex,assignerIndex,distance)

def calculate():
    """
    The main method
    Assigns points to clusters until all points are assigned by performing the calculations of the algorithm including the election, selection, and assignment operators 
    
    
    Parameters
    ----------    
    N/A
        
    Returns
    -------
    N/A
    """
    global distanceVectorIndex,nRemainingPoints,maxIndex,nElections
    global assignedPoints,points,nPoints,sortedDistancesTree
    global randomPointsIndices, randomPointsIndex, labelsPred
    global pointsNearestIndices, pointsNearestDistances
   
    
    randomPointsIndex = 0
    electedIndex = -1
    nearestIndex = -1
    
    #Assign points to clusters until all points are assigned (no points in remaining points list)
    while nRemainingPoints > 0:

        #Election: select the elected point (Elected)
        electedIndex = getElectedIndex()
        
        #make sure that Elected is already assigned to a cluster
        if electedIndex is not None:
            #Selection: select the next Nearest for the Elected
            nearestIndex,nearestDist = getNearestPoint(electedIndex)
            
            #In all cases the current index of the distance array should be incremented
            distanceVectorIndex[electedIndex] += 1
            
            #check if Nearest already clustered
            if isPointInCluster(nearestIndex):#The Nearest is already assigned to a cluster
                if not arePointsInSameCluster(electedIndex, nearestIndex): #The Nearest is already assigned to a different cluster than the Elected
                    #check if the Nearest should move to the cluster of the Elected
                    if shouldPointMoveToNearerCluster(nearestIndex, nearestDist): #The distance between the Nearest and the Assigner is greater than the distance between the Nearest and the Elected
                        moveNearestToElectedCluster(electedIndex, nearestIndex, nearestDist) #Assignmenet: The Nearest reassigned to the cluster of the Elected
            else: #The Nearest is not yet assigned to a cluster
                addNearestToElectedCluster(electedIndex, nearestIndex, nearestDist); #Assignmenet: Assign the Nearest to the cluster of the Elected

def getElectedIndex():
    """
    The Election Operation: Select the next point from the chromosome if exists. If not, randomly elect a point from the set of clustered points and add it to the chromosome. Name the point as Elected (E)
    
    Parameters
    ----------    
    N/A
        
    Returns
    -------
    int
        electedIndex: the index of the Elected
    """
    global randomPointsIndices, randomPointsIndex
    
    #Check if all genes of the chromosome has been Elected
    if len(randomPointsIndices) == randomPointsIndex:
        #select a random point that is already assigned to a cluster and add it to the chromosome
        electedIndex = getRandomAssignedPoint()

        if distanceVectorIndex[electedIndex] >= nPoints:
            #False Elected in case all points are selected as Nearest to the ELected (Which probably should not happen)
            return None
        else: 
            #add elected to chromosome
            randomPointsIndices.append(electedIndex)
            randomPointsIndex += 1
            return electedIndex            
    else: 
        #select the Elected from the chromosome if exists
        electedIndex = randomPointsIndices[randomPointsIndex]

        if not isPointInCluster(electedIndex):
            #False Elected in case it is not yet clustered (this might hapen due to crossover and mutation evolutionary operations)
            del randomPointsIndices[randomPointsIndex]
            return None            
        elif distanceVectorIndex[electedIndex] >= nPoints:
            #False Elected in case all points are selected as Nearest to the ELected (Which probably should not happen)
            del randomPointsIndices[randomPointsIndex]
            return None
        else:    
            #keep elected in chromosome
            randomPointsIndex += 1
            return electedIndex
   

        
def getRandomAssignedPoint():
    """
    select a random point that is already assigned to a cluster

    Parameters
    ----------    
    N/A
    
    Returns
    -------
    int
        The index of the random assigned point
    """
    global assignedPoints,nAssignedPoints
    return assignedPoints[random.randint(0,nAssignedPoints - 1)]
  
def getNearestPoint(electedIndex):
    """
    The Selection operation: Select the next nearest point for the Elected

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    
    Returns
    -------
    int
        The index of the Nearest
    float
        The distance between the Nearest and the Elected
    """
    global points,distanceVectorIndex,sortedDistancesTree
    point = points[electedIndex] #The Elected point
    distanceVectorIndexArray = distanceVectorIndex[electedIndex] #the distance vector index        
            
    distance,index = getNearestIndexAndDistance(electedIndex, point, distanceVectorIndexArray) 
    
    return index,distance

def getNearestIndexAndDistance(pointIndex, point,distanceVectorIndexArray):
    """
    Returns the index and distance of the Nearest point according to the distance vector index

    Parameters
    ----------    
    point : ndarray
        The point that we need to find its Nearest
    pointIndex : int
        The index of the point that we need to find its Nearest
    distanceVectorIndexArray : int
        The distance vector index which indicates the number of Nearest points previously selected by Elected
    
    Returns
    -------
    int
        The index of the Nearest
    float
        The distance between the point and the Nearest
    """
        
    global pointsNearestIndices, pointsNearestDistances, nPoints
    
    #Get the Nearest index and distance from pointsNearestIndices and pointsNearestDistances if exists
    distanceVectorIndexTree = distanceVectorIndexArray + 1
    if distanceVectorIndexArray < len(pointsNearestIndices[pointIndex]):
        index = pointsNearestIndices[pointIndex][distanceVectorIndexArray]
        distance = pointsNearestDistances[pointIndex][distanceVectorIndexArray]
        
    else:
        #Get the Nearest index and distance according to the current index of the distance vector        
        # p = 2 is the euclidean distance
        # k here is for how many nearest neighbors to return
    
        nearestDist,nearestIndex = sortedDistancesTree.query(point,p=2,k=[distanceVectorIndexTree]) 
        index = nearestIndex[0] #The Nearest index
        distance = nearestDist[0] #The distance between the Nearest and the Elected
        pointsNearestIndices[pointIndex].extend(nearestIndex)
        pointsNearestDistances[pointIndex].extend(nearestDist)
                    
    return distance,index

         
def isPointInCluster(pointIndex):
    """
    Checks if point is already assigned to a cluster

    Parameters
    ----------    
    pointIndex : int
        The index of the point to be checked
    
    Returns
    -------
    bool
        true/false indicating if the point is already assigned to a cluster        
    """
    global labelsPred    
    return labelsPred[pointIndex] != None


def arePointsInSameCluster(pointIndex1, pointIndex2):
    """
    Checks if two points are assigned to the same cluster

    Parameters
    ----------    
    pointIndex1 : int
        The index of the first point to be checked
    pointIndex2: int
        The index of the second point to be checked
    
    Returns
    -------
    bool
        true/false indicating if the two points are assigned to the same cluster        
    """
    global labelsPred
    return labelsPred[pointIndex1] == labelsPred[pointIndex2]
    
def shouldPointMoveToNearerCluster(nearestIndex, nearestDist): 
    """
    Checks if the Nearest should move to the cluster of the Elected

    Parameters
    ----------    
    nearestIndex : int
        The index of the Nearest point
    nearestDist: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    bool
        true/false if the Nearest should move to the cluster of the Elected        
    """
    global assignersDistances,assignersIndices
    if assignersIndices[nearestIndex] == None: #No assigner for the Nearest
        return False #Nearest should not move to the cluster of the Elected
    else:
        return assignersDistances[nearestIndex] > nearestDist #Nearest is closer to Elected than Assigner
    
def moveNearestToElectedCluster(electedIndex, nearestIndex, distance): 
    """
    Assignment operation: reassign the Nearest to the cluster of the Elected. 
    The Nearest is already assigned

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    nearestIndex: int
        The index of the Nearest
    distance: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    N/A
    """
    
    global labelsPred,nAssignedPoints,nRemainingPoints
    global assignedPoints, remainingPoints
    
    clusterNo = labelsPred[electedIndex] 
    oldAssignerIndex = assignersIndices[nearestIndex]
    
    if oldAssignerIndex != -1:
        #Change the cluster of the Nearest to the cluster of the Elected
        #Point is considered removed before adding it again
        nAssignedPoints -= 1
        nRemainingPoints += 1    
        assignedPoints = np.setdiff1d(assignedPoints, nearestIndex)
        remainingPoints = np.append(remainingPoints, nearestIndex) 
        addPointToCluster(clusterNo,nearestIndex,electedIndex,distance)
    
def addNearestToElectedCluster(electedIndex, nearestIndex, distance): # The Nearest is assigned to the cluster of the elected
    """
    Assignment operation: aSSIGN the Nearest to the cluster of the Elected.
    The Nearest is not yet assigned

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    nearestIndex: int
        The index of the Nearest
    distance: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    N/A
    """
    global labelsPred
    clusterNo = labelsPred[electedIndex]
    addPointToCluster(clusterNo,nearestIndex,electedIndex,distance)
    
def addPointToCluster(clusterNo,pointIndex,assignerIndex,assignerDistance):
    """
    Adds a point to a cluster

    Parameters
    ----------
    clusterNo : int
        The cluster where the point should be added
    pointIndex : int
        The index of the point to be added
    assignerIndex: int
        The index of the Assigner point for the point to be added
    assignerDistance: float
        The distance between the point and the Assigner
    Returns
    -------
    N/A
    """
    global remainingPoints,assignedPoints, assignmentsTree
    global nAssignedPoints,nRemainingPoints
    global assignersDistances,assignersIndices,labelsPred 

    nAssignedPoints += 1
    nRemainingPoints -= 1  
    
    #add point to assigned and remove from remaining
    assignedPoints = np.append(assignedPoints, pointIndex)
    remainingPoints = np.setdiff1d(remainingPoints, pointIndex)   
    #change the cluster of the point
    labelsPred[pointIndex] = clusterNo
    #Mark the Assigner to the point
    assignersIndices[pointIndex] = assignerIndex
    assignersDistances[pointIndex] = assignerDistance
    
    #Update the assignement tree
    if(assignmentsTree.contains(pointIndex)):
        children=assignmentsTree.subtree(pointIndex).all_nodes_itr()
        childrenIndices = np.array([child.identifier for child in children])
        labelsPred = np.array(labelsPred)
        labelsPred[childrenIndices] = clusterNo
        
    updateAssignmentsTree(pointIndex,assignerIndex)
    

def updateAssignmentsTree(pointIndex,assignerIndex):
    """
    Updates the assignment tree. The assignment tree contains the points that 
    are already assigned and their assigners and children in a tree data structure

    Parameters
    ----------    
    pointIndex : int
        The index of the point to be updated/added
    assignerIndex: int
        The index of the Assigner point for the point to be updated/added
    
    Returns
    -------
    N/A
    """
    global k,nAssignedPoints,nRemainingPoints,assignedPoints
    
    if assignerIndex == -1:
        assignerIndex = "root"
    
    if assignmentsTree.contains(pointIndex): # Point already assigned 
        pointTree = assignmentsTree.subtree(pointIndex)
        assignmentsTree.remove_node(pointIndex)        
        assignmentsTree.paste(assignerIndex, pointTree)                
    else:  # Point are not yet assigned 
        assignmentsTree.create_node(pointIndex, pointIndex, parent=assignerIndex)        
                
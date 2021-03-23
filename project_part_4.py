import numpy as np
import matplotlib as plt
from FEM_1D_Functions import *
from Second_Order_BVP import *

# configure solver inputs
a = 0
b = 1
L = 1.0
u_a = 0
u_b = 1.0
alpha = 5.0

domain = [a, b]
degree = 1
size = .1

#Creating mesh node array
[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)

#Creating Element Connectivity
connectivity = generateMeshConnectivity(numElements, degree)

#Calculate the gauss quadrature order
Nq = int(np.ceil(((degree+1)/2)))

#Define the function for the loading
def poissonF(x):

	fx = x

	return fx

#Compute global stiffness matrix
K = assembleGlobalStiffness(nodes, connectivity, Nq, degree)
print('K=', K)

#Compute global loading vector
F = assembleGlobalLoading(poissonF, nodes, connectivity, Nq, degree)
print('F=', F)

#Define the Dirichlet nodes
DirNodes = np.array([0, numNodes-1])

#Define the Dirichlet values
DirVals = np.array([u_a, u_b])

#Apply the Dirichlet boundaries 
[Kg, Fg] = applyDirichlet(K, F, DirNodes, DirVals)

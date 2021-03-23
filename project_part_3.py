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
degree = 2
size = .1
elementNumber = 0 #Input element number, starting from 0. 

#Creating mesh node array
[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)

#Creating Element Connectivity
connectivity = generateMeshConnectivity(numElements, degree)

#Calculate the gauss quadrature order
Nq = int(np.ceil(((degree+1)/2)))

#Call the element stiffness computation 
k = computeElementStiffness(nodes[connectivity[elementNumber]], Nq, degree)

print('kij=', k)

#Define the function for the loading 
def poissonF(x):

	fx = x

	return fx

#Call the element loading computation
f = computeElementLoading(poissonF, nodes[connectivity[elementNumber]], Nq, degree)

print('fi=', f)







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
elementNumber = 0 #Input element number, starting from 0. 

[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)
connectivity = generateMeshConnectivity(numElements, degree)
print('Number of Elements=', numElements)
Nq = int(np.ceil(((degree+1)/2)))


k = computeElementStiffness(nodes[connectivity[elementNumber]], Nq, degree)

print('kij=', k)

def poissonF(x):

	#f = k**2 * cos((np.pi*k*x)/L) + alpha*(1-k**2)*sin((2*np.pi*k*x)/L)
	fx = x

	return fx



f = computeElementLoading(poissonF, nodes[connectivity[elementNumber]], Nq, degree)

print('fi=', f)







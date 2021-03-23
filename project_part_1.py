import numpy as np

from FEM_1D_Functions import *

# configure solver inputs
a = 0
b = 1
L = 1.0
u_a = 0
u_b = 1.0
alpha = 5.0

domain = [a, b]
degree = 3
size = 0.1

#4.a. Creating mesh node array
[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)

#Print out the results
print("Number of Elements:", numElements)
print("Number of Nodes:", numNodes)
print("List of Nodes:", nodes)

#4.b. Creating Element Connectivity
connectivity = generateMeshConnectivity(numElements, degree)

print("Connectivity Matrix:", connectivity)
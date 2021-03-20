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
size = .5


[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)
connectivity = generateMeshConnectivity(numElements, degree)
print('nodes=', nodes)
print('index=', range(len(nodes)-1))
Nq = int(np.ceil(((degree+1)/2)))
print('Nq=', Nq)
def poissonF(x):

	#f = k**2 * cos((np.pi*k*x)/L) + alpha*(1-k**2)*sin((2*np.pi*k*x)/L)
	fx = x

	return fx


K = assembleGlobalStiffness(nodes, connectivity, Nq, degree)
print('K=', K)

F = assembleGlobalLoading(poissonF, nodes, connectivity, Nq, degree)
print('F=', F)

DirNodes = np.array([np.where(nodes == a)[0][0], np.where(nodes == b)[0][0]])

DirVals = np.array([u_a, u_b])

print(DirNodes)
print(DirVals)

for i in range(len(DirNodes)):
	[Kg, Fg] = applyDirichlet(K, F, DirNodes[i], DirVals[i])

	print('Kg=', Kg)
	print('Fg=', Fg)
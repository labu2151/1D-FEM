import numpy as np
import matplotlib as plt
from FEM_1D_Functions import *
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

[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)
connectivity = generateMeshConnectivity(numElements, degree)

eta = np.linspace(-1,1,100)

refNodes = referenceElementNodes(degree)

localNodes = range(len(refNodes))

basis = np.zeros((len(localNodes), len(eta)))

plt.figure()
for i in range(len(localNodes)):
    for j in range(len(eta)):
        basis[i,j] = lineElementShapeFunction(eta[j], degree, localNodes[i])
    plt.plot(eta, basis[i,:])
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('Shape Functions')
    plt.savefig('shape_functions.png')


shapeDiff = np.zeros((len(localNodes), len(eta)))

plt.figure()
for i in range(len(localNodes)):
    for j in range(len(eta)):
        shapeDiff[i,j] = lineElementShapeDerivatives(eta[j], degree, localNodes[i])
    plt.plot(eta, shapeDiff[i,:])
    plt.xlabel('eta')
    plt.ylabel('dphi/deta')
    plt.title('Shape Function Derivatives')
    plt.savefig('shape_Derivatives.png')


Nq = int(np.ceil(((degree+1)/2)))
print('Nq =', Nq)
eta_gauss = getGaussQuadraturePoints(Nq)
print('eta_gauss=', eta_gauss)
isomap = np.zeros(len(eta_gauss))

for i in range(numElements):
	index = connectivity[i,:]
	coord = nodes[index] 
	for j in range(Nq): 
		isomap[j] = lineElementIsoparametricMap(coord, degree, eta_gauss[j])

print('isomap=', isomap)


dXdEta = np.zeros(len(eta_gauss))
for i in range(numElements):
    index = connectivity[i,:]
    coord = nodes[index]
    for j in range(Nq):
        dXdEta[j] = lineElementMappingGradient(coord, degree, eta_gauss[j])

print('dXdEta=', dXdEta)

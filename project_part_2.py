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
degree = 4
size = .1

#Creating mesh node array
[numElements, numNodes, nodes] = generateMeshNodes(domain, degree, size)

#Creating Element Connectivity
connectivity = generateMeshConnectivity(numElements, degree)

#Create an array of eta values to be passed into the functions (for plotting)
eta = np.linspace(-1,1,100)

#5.a. Computing nodes for reference element of a given polynomial order
refNodes = referenceElementNodes(degree)

#Create variable localNodes which is essentially a counter of the reference nodes
localNodes = range(len(refNodes))

#5.b. Element Shape functions and their derivatives
#Initialize the basis value vector
basis = np.zeros((len(localNodes), len(eta)))

#Calculate and plot the shape functions
plt.figure()
for i in range(len(localNodes)): #for each local node
    for j in range(len(eta)): #for each eta value
        basis[i,j] = lineElementShapeFunction(eta[j], degree, localNodes[i])
    plt.plot(eta, basis[i,:])
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('4th Order Polynomial Element Shape Functions')
    plt.savefig('4th_shape_functions.png')

#Initialize the shape function derivative vector
shapeDiff = np.zeros((len(localNodes), len(eta)))

#Calculate and plot the shape derivatives
plt.figure()
for i in range(len(localNodes)): #for each local node
    for j in range(len(eta)): #for each eta value
        shapeDiff[i,j] = lineElementShapeDerivatives(eta[j], degree, localNodes[i])
    plt.plot(eta, shapeDiff[i,:])
    plt.xlabel('eta')
    plt.ylabel('dphi/deta')
    plt.title('4th Order Polynomial Element Shape Function Derivatives')
    plt.savefig('4th_shape_Derivatives.png')

#5.c. Isoparametric mapping and its gradient
#Calculate the gauss quadrature order
Nq = int(np.ceil(((degree+1)/2)))

#Get the gauss quadrature points corresponding to the gauss order
eta_gauss = getGaussQuadraturePoints(Nq)

#Initialize the isomap vector
isomap = np.zeros(len(eta_gauss))

#Calculate the isoparametric mapping
for i in range(numElements): #for the number of elements
	index = connectivity[i,:] #get the row from the connectivity matrix
	coord = nodes[index]  #Get the nodes corresponding the the row from the connectivity matrix
	for j in range(Nq):  #For each gauss point 
		isomap[j] = lineElementIsoparametricMap(coord, degree, eta_gauss[j])

#Calculate the isoparamtric mapping gradient
#Initialize the gradient vector
dXdEta = np.zeros(len(eta_gauss))
for i in range(numElements): #for the number of elements
    index = connectivity[i,:] #get the row from the connectivity matrix
    coord = nodes[index] #Get the nodes corresponding the the row from the connectivity matrix
    for j in range(Nq): #For each gauss point
        dXdEta[j] = lineElementMappingGradient(coord, degree, eta_gauss[j])


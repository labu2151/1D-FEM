import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.sparse.linalg import cg
except ImportError:
    sys.exit('Scientific Python Stack: NumPy+SciPy+MatPlotLib Not Installed!')

try:
    from FEM_1D_Functions import *
except ImportError:
    sys.exit('Functions From FEM_1D_Functions Not Successfully Loaded!')

try:
    from Second_Order_BVP import *
except ImportError:
    sys.exit('Functions From Second_Order_BVP Not Successfully Loaded!')

#------------------------
# configure solver inputs
#------------------------

x0 = 0
x1 = 1
p = 1 #Polynomial Degree
h = .05 #Mesh sizing parameter
gN = int(np.ceil(((p+1)/2))) #Gauss order

fun = lambda x: (2**2)*np.cos(np.pi*2*x/1.0) + 5.0*(1-2**2)*np.sin(2*np.pi*2*x/1.0)

[numElements, numNodes, nodes] = generateMeshNodes([x0, x1], p, h, a_ReturnNumElements=True, a_ReturnNumNodes=False)

E = generateMeshConnectivity(numElements, p)

K = assembleGlobalStiffness(nodes, E, gN, p)

F = assembleGlobalLoading(fun, nodes, E, gN, p)

dirNodes = np.array([np.where(nodes == x0)[0][0], np.where(nodes == x1)[0][0]])

dirVals = np.array([0, 1])

for i in range(len(dirNodes)):
	[Kg, Fg] = applyDirichlet(K, F, dirNodes[i], dirVals[i])


[uSol, status] = cg(Kg, Fg)

plt.figure()
plt.plot(nodes, uSol)
plt.xlabel('Nodal Coordinates')
plt.ylabel('Solution Values')
plt.savefig('sol1.png')
#-----------------------------------------
# define the functionS for f
# and the exact solution you have computed
#------------------------------------------


#--------------------------------------------------------
# use the finite element functions to obtain the solution
#--------------------------------------------------------

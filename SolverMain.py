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
u_a = 0.0
u_b = 1.0
p = 1 #Polynomial Degree
h = 0.005 #Mesh sizing parameter
k = 32.0
alpha = 5.0
L = 1.0
gN = int(np.ceil(((p+1)/2))) #Gauss order

#fun = lambda x: -1.0*((k**2)*np.cos(np.pi*k*x/L) + alpha*(1-k**2)*np.sin(2*np.pi*k*x/L))
fun = lambda x: -1.0*(k**2*np.cos((np.pi*k*x)/(L)) + alpha*(1-k**2)*np.sin((2*np.pi*k*x)/(L)))

[numElements, numNodes, nodes] = generateMeshNodes([x0, x1], p, h, a_ReturnNumElements=True, a_ReturnNumNodes=False)

E = generateMeshConnectivity(numElements, p)

K = assembleGlobalStiffness(nodes, E, gN, p)

F = assembleGlobalLoading(fun, nodes, E, gN, p)

dirNodes = np.array([0, numNodes-1])

dirVals = np.array([u_a, u_b])


[Kg, Fg] = applyDirichlet(K, F, dirNodes, dirVals)

#--------------------------------------------------------
# use the finite element functions to obtain the solution
#--------------------------------------------------------

[uSol, status] = cg(Kg, Fg)



#-----------------------------------------
# define the functionS for f
# and the exact solution you have computed
#------------------------------------------

u_exact = ((L**2/np.pi**2)*np.cos(np.pi*k/L) + ((alpha*(1-k**2)*L**2)/(4*np.pi**2*k**2))*np.sin(2*np.pi*k/L) - (L**2/np.pi**2) + 1)*nodes \
		-(L**2/np.pi**2)*np.cos(np.pi*k*nodes/L) - ((alpha*(1-k**2)*L**2)/(4*np.pi**2*k**2))*np.sin(2*np.pi*k*nodes/L) + (L**2/np.pi**2)


#Plotting
plt.figure()
plt.plot(nodes, u_exact, 'b', label='Exact Solution')
plt.plot(nodes, uSol, 'r--', label='FEM Solution')
plt.xlabel('Nodal Coordinates')
plt.ylabel('Solution Values')
plt.title('Solution for k=32.0, h=0.005')
plt.legend()
plt.grid()
#plt.savefig('sol20.png')


#Error calculation
error = np.sum(np.absolute(uSol - u_exact))

print('Error=', error)
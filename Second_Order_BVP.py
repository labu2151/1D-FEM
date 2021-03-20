'''Solver for a 1D Model Boundary Value Problem

The 1-D FEM functions are used to solve here a model BVP of the following form

d^2 u/dx^2 = f(x) \forall x \in [a, b]
u = u_a at x = a
u = u_b at x = b

Developed for computational fluid dynamics class taught at the Paul M Rady
Department of Mechanical Engineering at the University of Colorado Boulder by
Prof. Debanjan Mukherjee.

All inquiries addressed to Prof. Mukherjee directly at debanjan@Colorado.Edu

'''

import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit('Scientific Python Stack: NumPy+SciPy+MatPlotLib Not Installed!')

try:
    from FEM_1D_Functions import *
except ImportError:
    sys.exit('Functions From FEM_1D_Functions Not Successfully Loaded!')


def computeElementStiffness(a_ElementNodes, a_GaussOrder, a_Degree):
    """Compute element stiffness matrix for a simple second order BVP

    The boundary value problem involves the differential equation:
    d^2 u
    ----- = f(x)
    dx^2

    Args:
        a_ElementNodes (float-array): the nodal coordinates (global) of the element
        a_GaussOrder (int): the order of Gaussian quadrature needed
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        k: a float-array holding the computed stiffness matrix for the element

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    eta_gauss = getGaussQuadraturePoints(a_GaussOrder)
    w = getGaussQuadratureWeights(a_GaussOrder)

    refNodes = referenceElementNodes(a_Degree)

    localNodes = range(len(refNodes))
    
    K = np.zeros((len(localNodes), len(localNodes)))
    for i in range(len(localNodes)):
        for j in range(len(localNodes)):
            for k in range(a_GaussOrder):
                dXdEta = lineElementMappingGradient(a_ElementNodes, a_Degree, eta_gauss[k])
                phiDiffi = lineElementShapeDerivatives(eta_gauss[k], a_Degree, localNodes[i])
                phiDiffj = lineElementShapeDerivatives(eta_gauss[k], a_Degree, localNodes[j])

                K[i,j] = K[i,j] + w[k]*phiDiffi*phiDiffj*(1/dXdEta)

    return K  
    
    ##


def computeElementLoading(a_Function, a_ElementNodes, a_GaussOrder, a_Degree):
    """Compute element rhs loading vector for a simple second order BVP

    The boundary value problem involves the differential equation:
    d^2 u
    ----- = f(x)
    dx^2

    Args:
        a_Function (function): the handle to the function f(x) (implemented in the solver)
        a_ElementNodes (float-array): the nodal coordinates (global) of the element
        a_GaussOrder (int): the order of Gaussian quadrature needed
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        f: a float-array holding the computed loading vector for the element

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """
    eta_gauss = getGaussQuadraturePoints(a_GaussOrder)
    w = getGaussQuadratureWeights(a_GaussOrder)

    refNodes = referenceElementNodes(a_Degree)

    localNodes = range(len(refNodes))
  
    f = np.zeros(len(localNodes))

    for i in range(len(localNodes)):
        for j in range(a_GaussOrder):
            dXdEta = lineElementMappingGradient(a_ElementNodes, a_Degree, eta_gauss[j])
            isomap = lineElementIsoparametricMap(a_ElementNodes, a_Degree, eta_gauss[j])
            phi = lineElementShapeFunction(eta_gauss[j], a_Degree, localNodes[i])

          
            f[i] = f[i] + w[j]*phi*a_Function(isomap)*(dXdEta)
   
    return f



def assembleGlobalStiffness(a_Nodes, a_Connectivity, a_GaussOrder, a_Degree):
    """Assemble the element stiffness matrices into a global stiffness matrix

    This is to obtain the assembled matrix over the entire specified mesh
    for the boundary value problem:
    d^2 u
    ----- = f(x)
    dx^2

    Args:
        a_Nodes (float-array): element nodal coordinates for the entire mesh
        a_Connectivity (int-array): element connectivity matrix for the mesh
        a_GaussOrder (int): the order of Gaussian quadrature needed
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        k_g: float array containing the assembled global stiffness matrix

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """
    
    numElements = len(a_Connectivity)
    
    numElementNodes = np.shape(a_Connectivity)[1]
    
    K = np.zeros((len(a_Nodes), len(a_Nodes)))

    for e in range(numElements):
        index = a_Connectivity[e,:]
        coord = a_Nodes[index] 
        ke = computeElementStiffness(coord, a_GaussOrder, a_Degree)
        
        for i in range(numElementNodes):
            
            for j in range(numElementNodes):
                
                K[a_Connectivity[e,i], a_Connectivity[e,j]] += ke[i,j]    
    return K


def assembleGlobalLoading(a_Function, a_Nodes, a_Connectivity, a_GaussOrder, a_Degree):
    """Assemble the element rhs loading vectors into a global rhs vector

    This is to obtain the assembled matrix over the entire specified mesh
    for the boundary value problem:
    d^2 u
    ----- = f(x)
    dx^2

    Args:
        a_Function (function): the handle to the function f(x) (implemented in the solver)
        a_Nodes (float-array): element nodal coordinates for the entire mesh
        a_Connectivity (int-array): element connectivity matrix for the mesh
        a_GaussOrder (int): the order of Gaussian quadrature needed
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        f_g: float array containing the assembled global rhs loading vector

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """
    numElements = len(a_Connectivity)
    
    numElementNodes = np.shape(a_Connectivity)[1]
    F = np.zeros(len(a_Nodes))
    
    for e in range(numElements):
        index = a_Connectivity[e,:]
        coord = a_Nodes[index]
        
        fe = computeElementLoading(a_Function, coord, a_GaussOrder, a_Degree)
        
        for i in range(numElementNodes):
            
            F[a_Connectivity[e,i]] += fe[i]

    return F
  


def applyDirichlet(a_Kg, a_Fg, a_DirNodes, a_DirVals):
    """Modify the matrix system by applying the Dirichlet boundary conditions

    Args:
        a_Kg (float-array): global stiffness matrix
        a_Fg (float-array): global rhs loading vector
        a_DirNodes (int): list of node IDs where Dirichlet values are specified
        a_DirVals (float): list of Dirichlet values

    Returns:
        a_Kg: the modified global stiffness matrix
        a_Fg: the modified global stiffness matrix

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """
    i = a_DirNodes
    u0 = a_DirVals
    a_Kg[i,:] = 0.0

    a_Fg = a_Fg - a_Kg[:,i]*u0

    a_Kg[:,i] = 0.0

    a_Kg[i,i] = 1.0

    a_Fg[i] = u0

    return a_Kg, a_Fg

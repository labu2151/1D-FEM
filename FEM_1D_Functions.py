'''Implementation of 1D FEM Functions

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


def generateMeshNodes(a_Domain, a_Degree, a_Size, a_ReturnNumElements=True, a_ReturnNumNodes=False):
    """Generate the node coordinates array for the 1-dimensional mesh

    Args:
        a_Domain (2X1 list): the lower bound and upper bound of linear domain
        a_Degree (int): the polynomial degree of the finite element/interpolation
        a_Size (float): the size of each element in the mesh
        a_ReturnNumElements (bool): set to True if number of elements is to be returned
        a_ReturnNumNodes (bool): set to True if number of nodes is to be returned

    Returns:
        nodes (float array): numpy array with nodal coordinates
        numNodes (int, optional): number of nodes (if a_ReturnNumNodes = True)
        numElements (int, optional): number of elements (if a_ReturnNumElements = True)

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    a = a_Domain[0]
    b = a_Domain[1]

    nodes = np.linspace(a, b, int(b/a_Size)*a_Degree + 1)
    
    numNodes = len(nodes)
    numElements = int((numNodes-1)/a_Degree)

    return [numElements, numNodes, nodes]
    ##



def generateMeshConnectivity(a_NumElements, a_Degree):
    """Generate the mesh element connectivity matrix for the 1-dimensional mesh

    Args:
        a_NumElements (int): the number of elements in the mesh
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        connectivity: NumPy array of ints of size a_NumElements X (a_Degree+1)

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    #create an array for the node id (dependent on the number of elements and the order)
    node_id = np.linspace(0, a_NumElements*a_Degree, a_NumElements*a_Degree + 1)
    
    #Initialize the connectivity matrix
    connectivity = np.zeros((a_NumElements, a_Degree+1))

    #Create the connectivity matrix
    for i in range(a_NumElements):
        connectivity[i,:] = node_id[a_Degree*i:a_Degree*i+(a_Degree+1)]
    
    connectivity = connectivity.astype(int)
    
    return connectivity
    ##



def referenceElementNodes(a_Degree):
    """Generate nodal coordinates for the 1-dimensional reference element in [-1.0,1.0]

    Args:
        a_Degree (int): the polynomial degree of the finite element/interpolation

    Returns:
        refNodes: NumPy float array of reference element node locations (in [-1.0,1.0])

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    refNodes = np.linspace(-1,1, a_Degree+1)
    
    return refNodes

    ##


def lineElementShapeFunction(a_Eta, a_Degree, a_LocalNode):
    """Compute the shape function value for a line element

    Args:
        a_Eta (float): coordinate in the local coordinate system where shape function is evaluated
        a_Degree (int): the polynomial degree of the finite element/interpolation
        a_LocalNode (int): ID of the local node for which the shape function is evaluated

    Returns:
        basisVal: single float scalar value of the shape function evaluated in local system

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    refNode = referenceElementNodes(a_Degree)
    phi = np.ones((a_Degree+1, 1))
    for i in range(a_Degree+1):
        for k in range(a_Degree+1):
            if i != k:
                phi[i,:] = phi[i,:]*(a_Eta - refNode[k])/(refNode[i] - refNode[k])
                
    basisVal = float(phi[a_LocalNode, :])
  
    return basisVal
    ##


def lineElementShapeDerivatives(a_Eta, a_Degree, a_LocalNode):
    """Compute the shape function derivatives for a line element

    Args:
        a_Eta (float): coordinate in the local coordinate system where shape function derivative is evaluated
        a_Degree (int): the polynomial degree of the finite element/interpolation
        a_LocalNode (int): ID of the local node for which the shape function is evaluated

    Returns:
        val: single float scalar value of the shape function derivative evaluated in local system

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    refNode = referenceElementNodes(a_Degree)
    
    prod = 1
    
    phiDiff = np.zeros((a_Degree+1, 1))
    
    for i in range(a_Degree+1):
        for k in range(a_Degree+1):
            if i != k:
                diff1 = 1/(refNode[i] - refNode[k])      
                for j in range(a_Degree+1):
                    if j != k and j!=i:
                        prod = prod*((a_Eta - refNode[j])/(refNode[i] - refNode[j]))
                       
                phiDiff[i,:] = phiDiff[i,:] + (diff1*prod)
                
                prod = 1
                
    
    shapeDiffVal = float(phiDiff[a_LocalNode, :])
    
    return shapeDiffVal
    ##


def lineElementIsoparametricMap(a_ElementNodeCoordinates, a_Degree, a_Eta):
    """Evaluate the isoparametric mapping of global-local coordinates

    Args:
        a_ElementNodeCoordinates (float array): element nodal coordinates (in global system)
        a_Degree (int): the polynomial degree of the finite element/interpolation
        a_Eta (float): coordinate in the local system where mapping is evaluated

    Returns:
        isoMap: mapped coodinate value (float) using shape function interpolation

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    refNodes = referenceElementNodes(a_Degree)

    localNodes = range(len(refNodes))
    isoMap = 0
    for i in range(len(localNodes)):
        phi = lineElementShapeFunction(a_Eta, a_Degree, localNodes[i])

        isoMap = isoMap + phi*a_ElementNodeCoordinates[i]
    
    
    return float(isoMap)


def lineElementMappingGradient(a_ElementNodeCoordinates, a_Degree, a_Eta):
    """Evaluate the Jacobian of the isoparametric mapping of global-local coordinates

    Args:
        a_ElementNodeCoordinates (float array): element nodal coordinates (in global system)
        a_Degree (int): the polynomial degree of the finite element/interpolation
        a_Eta (float): coordinate in the local system where mapping is evaluated

    Returns:
        dXdEta: mapped coodinate value (float) using shape function interpolation

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    ##
    refNodes = referenceElementNodes(a_Degree)

    localNodes = range(len(refNodes))
    dXdEta = 0
    for i in range(len(localNodes)):
        phiDiff = lineElementShapeDerivatives(a_Eta, a_Degree, localNodes[i])

        dXdEta = dXdEta + phiDiff*a_ElementNodeCoordinates[i]
    
    
    return float(dXdEta)
    ##


def plotMesh(a_Nodes, a_Connectivity, a_XLabel=None, a_YLabel=None, a_SaveFigure=None):
    """Create a plot of the linear mesh.

    Internal nodes are painted in blue triangles. Edge nodes are painted in red squares

    Args:
        a_Nodes (float-array): element nodal coordinates (in global system)
        a_Connectivity (int-array): element connectivitty matrix
        a_XLabel (str): set this to the text used for x-axis label
        a_YLabel (str): set this to the text used for y-axis label
        a_SaveFigure (str): set this to the name of the file to store the figure

    Returns:
        none

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    if a_Connectivity.shape[1] == 2:

        plt.plot(a_Nodes, np.zeros_like(a_Nodes), 'rs-')

    else:
        edgeNodes   = np.unique(np.concatenate((a_Nodes[a_Connectivity[:,0]], a_Nodes[a_Connectivity[:,-1]])))
        inNodes     = a_Nodes[np.array([x not in edgeNodes for x in a_Nodes])]
        print(inNodes)
        plt.plot(edgeNodes, np.zeros_like(edgeNodes), 'rs-')
        plt.plot(inNodes, np.zeros_like(inNodes), 'bv')

    plt.show()


def plotSolutions(a_Nodes, a_Connectivity, a_Solution, a_YLabel=None, a_XLabel=None, a_SaveFigure=None):
    """Create a plot of the solution over the linear mesh

    Internal nodes are painted in blue triangles. Edge nodes are painted in red squares/.
    Solutiom data are painted in magenta squares.

    Args:
        a_Nodes (float-array): element nodal coordinates (in global system)
        a_Connectivity (int-array): element connectivitty matrix
        a_Solution (float-array): element nodal solutions
        a_XLabel (str): set this to the text used for x-axis label
        a_YLabel (str): set this to the text used for y-axis label
        a_SaveFigure (str): set this to the name of the file to store the figure

    Returns:
        none

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    if a_Connectivity.shape[1] == 2:

        plt.plot(a_Nodes, np.zeros_like(a_Nodes), 'rs-')

    else:

        edgeNodes   = np.unique(np.concatenate((a_Nodes[a_Connectivity[:,0]], a_Nodes[a_Connectivity[:,-1]])))
        inNodes     = a_Nodes[np.array([x not in edgeNodes for x in a_Nodes])]
        plt.plot(edgeNodes, np.zeros_like(edgeNodes), 'rs-')
        plt.plot(inNodes, np.zeros_like(inNodes), 'bv')

    plt.plot(a_Nodes, a_Solution, 'ms-')

    if a_YLabel is not None:
        plt.ylabel(a_YLabel, fontweight='bold')

    if a_XLabel is not None:
        plt.xlabel(a_XLabel, fontweight='bold')

    if a_SaveFigure is not None:
        plt.savefig(a_SaveFigure, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def getGaussQuadratureWeights(a_Degree):
    """Tabulation of weights for Gauss quadrature

    Args:
        a_Degree (int): the order of Gaussian quadrature needed

    Returns:
        float-array: the scalar weights for quadrature evaluation

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    if a_Degree == 1:
        return [2.0]
    elif a_Degree == 2:
        return [1.0, 1.0]
    elif a_Degree == 3:
        return [5.0/9.0, 8.0/9.0, 5.0/9.0]
    elif a_Degree == 4:
        return [0.652145, 0.347855, 0.652145, 0.347855]
    else:
        sys.exit("Only upto 4th degree Gauss quadrature is implemented")


def getGaussQuadraturePoints(a_Degree):
    """Tabulation of integration points for Gauss quadrature

    Points returned on domain [-1.0,1.0]

    Args:
        a_Degree (int): the order of Gaussian quadrature needed

    Returns:
        float-array: the scalar 1-dimensonal coordinates for quadrature evaluation

    .. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
    """

    if a_Degree == 1:
        return [0.0]
    elif a_Degree == 2:
        return [-0.57735, 0.57735]
    elif a_Degree == 3:
        return [-0.774597, 0.0, 0.774597]
    elif a_Degree == 4:
        return [-0.339981, -0.861136, 0.339981, 0.861136]
    else:
        sys.exit("Only upto 4th degree Gauss quadrature is implemented")

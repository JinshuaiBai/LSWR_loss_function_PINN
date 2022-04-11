"""

    This function is to calculate the Si for numerical integration. Here we adopt the Delaunay segementation
    algorithm provided by the SciPy package. Details of this algorithm can be found in: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
"""
import numpy as np
from scipy.spatial import Delaunay

def Tri_Segm(xy, ns):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'xy'        [Array of float32]      ns*2        : Coordinates of all the sample points;
        'ns'        [int]                   1           : Total number of sample points.
        
    Variables:
        Name        Type                    Size        Info.
        
        [ne]        [float]                 1           : Total number of triangle segementations;
        [buer]      [Array of float32]      ne*3        : A array that contains the indices of the points forming the simplices in the triangulation;
        [ar]        [Array of float32]      ns*1        : Si for each triangle segementations.

    =================================================================================================================================    
    """
    tri=Delaunay(xy)
    buer=tri.simplices    
    ne=len(buer)
    ar = np.zeros((ns, 1)).astype(np.float32)
    for i in range(0,ne):
        ar_t = xy[buer[i][0]][0] * xy[buer[i][1]][1] + \
            xy[buer[i][1]][0] * xy[buer[i][2]][1] + \
            xy[buer[i][2]][0] * xy[buer[i][0]][1] - \
            xy[buer[i][0]][0] * xy[buer[i][2]][1] - \
            xy[buer[i][1]][0] * xy[buer[i][0]][1] - \
            xy[buer[i][2]][0] * xy[buer[i][1]][1]
        ar[buer[i][0]] = ar[buer[i][0]] + ar_t/6
        ar[buer[i][1]] = ar[buer[i][1]] + ar_t/6
        ar[buer[i][2]] = ar[buer[i][2]] + ar_t/6
    return ar
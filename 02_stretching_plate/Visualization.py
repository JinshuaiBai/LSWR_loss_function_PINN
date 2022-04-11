"""

    This function is to visualize the displacment and stress contours.
        
"""
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def Vis(E, pinn, net1, net2, T, L, it):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'E'         [float]                 1           : Young's module;
        'pinn'      [keras model]           \           : The trained PINN;
        'net1'      [keras model]           \           : The trained FNN for displacment u;
        'net2'      [keras model]           \           : The trained FNN for displacment v;
        'xy'        [Array of float32]      ns*2        : The sample points coordinates for visualization.
        
    Variables:
        Name        Type                    Size        Info.
        
        [u]         [Array of float32]      ns*1        : Displacement u;
        [v]         [Array of float32]      ns*1        : Displacement v;
        [s11]       [Array of float32]      ns*1        : Nornal stress sigma_x;
        [s22]       [Array of float32]      ns*1        : Nornal stress sigma_y;
        [s12]       [Array of float32]      ns*1        : Shear stress tau_xy.

    =================================================================================================================================    
    """

    xy = np.zeros((201*201, 2)).astype(np.float32)
    k = 0
    for i in range(0,201):
        for j in range(0,201):
                xy[k, 0] = i * 1/200
                xy[k, 1] = j * 1/200
                k = k+1
                
    u = net1.predict(xy) * xy[..., 0, np.newaxis]
    v = net2.predict(xy) * xy[..., 1, np.newaxis]
    temp = pinn.predict([ xy for i in range(0,5) ])
    s11 = temp[6] * E
    s22 = temp[3] * E
    s12 = temp[2] * E
    
    # plot figure for displacement u
    fig1 = plt.figure(1)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = u, cmap = 'jet', vmin = 0, vmax = 0.015)
    plt.axis('equal')
    plt.colorbar()
    plt.title('v')
    
    # plot figure for displacement v
    fig2 = plt.figure(2)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = v, cmap = 'jet', vmin = -4e-3, vmax = 0)
    plt.axis('equal')
    plt.colorbar()
    plt.title('v')
    
    # plot figure for stress sigma_x
    fig3 = plt.figure(3)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s11, cmap = 'jet', vmin = 0, vmax = 1)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\sigma_{x}$')
    
    # plot figure for stress sigma_y
    fig4 = plt.figure(4)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s22, cmap = 'jet', vmin = -0.1, vmax = 0.4)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\sigma_{y}$')
    
    # plot figure for stress tau_xy
    fig5 = plt.figure(5)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = s12, cmap = 'jet', vmin = -0.1, vmax = 0)
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$\tau_{xy}$')
    
    # output data in the 'out.mat' file
    scipy.io.savemat('out.mat', {'xy': xy, 'u': np.hstack([u,v]), 's11': s11, 's22': s22, 's12': s12, 'T': T, 'L': L, 'it': it}) 
"""

    This function is for initialize all the variables for the calculation.
        
"""
import numpy as np

def Input():
    """
    =================================================================================================================================
    
    Variables:
        Name        Type                    Size        Info.
        
        [ns]        [int]                   1           : Total number of sample points;
        [ns_u]      [int]                   1           : Number of sample points on top boundary of the beam;
        [ns_l]      [int]                   1           : Number of sample points on left boundary of the beam;
        [dx]        [float]                 1           : Sample points interval;
        [h]         [float]                 1           : Parameter relates to sample points interval;
        [xy]        [Array of float32]      ns*2        : Coordinates of all the sample points;
        [xy_u]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the top boundary of the beam;
        [xy_b]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the bottom boundary of the beam;
        [xy_l]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the left boundary of the beam;
        [xy_r]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the right boundary of the beam;
        [x_train]   [List]                  5           : PINN input list, contains all the coordinates information;
        [s_u_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the top boundary of the beam;
        [s_u_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the top boundary of the beam;
        [s_b_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the bottom boundary of the beam;
        [s_b_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the bottom boundary of the beam;
        [s_l_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the left boundary of the beam;
        [s_l_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the left boundary of the beam;
        [s_r_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the right boundary of the beam;
        [s_r_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the right boundary of the beam;
        [y_train]   [List]                  8           : PINN boundary condition list, contains all the force boundary conditions;
        [E]         [float]                 1           : Young's module;
        [mu]        [float]                 1           : Poisson ratio.
        
    =================================================================================================================================    
    """
    print('-------------------------------------------------\n')
    print('                 Input Info.\n')
    print('-------------------------------------------------\n')
    
    # number of sample points
    ns = 561
    ns_u = 51
    ns_l = 11
    
    # sample points' interval
    dx = 0.01
    h = 100*dx
    
    print(ns, 'sample points')
    print(ns_u,'sample points on the top boundary;',ns_u,'sample points on the bottom boundary;')
    print(ns_l,'sample points on the left boundary;',ns_l,'sample points on the right boundary.')
    
    # initialize sample points' coordinates    
    xy = np.zeros((ns, 2)).astype(np.float32)
    for i in range(0,ns_u):
        for j in range(0,ns_l):
            xy[i*ns_l+j,0] = i * dx
            xy[i*ns_l+j,1] = j * dx - 0.05
    xy_u = np.hstack([np.linspace(0,0.5, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      0.05*np.ones((ns_u,1)).astype(np.float32)])
    xy_b = np.hstack([np.linspace(0,0.5, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      -0.05*np.ones((ns_u,1)).astype(np.float32)])
    xy_l = np.hstack([np.zeros((ns_l,1)).astype(np.float32), \
                  np.linspace(-0.05,0.05, ns_l).reshape(ns_l, 1).astype(np.float32)])
    xy_r = np.hstack([0.5*np.ones((ns_l,1)).astype(np.float32), \
                  np.linspace(-0.05,0.05, ns_l).reshape(ns_l, 1).astype(np.float32)])
    
    # create PINN input list
    x_train = [ xy, xy_u, xy_b, xy_l, xy_r ]
    
    # material properties
    E = 1000.
    mu = 0.3
    print('The Young''s module is', E,'; The Possion''s ratio is', mu,'.\n')
    print('-------------------------------------------------\n')
    
    # boundary conditions
    s_u_x = np.zeros((ns_u,1)).astype(np.float32)
    s_u_y = np.zeros((ns_u,1)).astype(np.float32)
    s_b_x = np.zeros((ns_u,1)).astype(np.float32)
    s_b_y = np.zeros((ns_u,1)).astype(np.float32)
    s_l_x = np.zeros((ns_l,1)).astype(np.float32)
    s_l_y = np.zeros((ns_l,1)).astype(np.float32)
    s_r_x = xy_r[:,1,np.newaxis]*1000
    s_r_y = np.zeros((ns_l,1)).astype(np.float32)
    
    # create PINN boundary condition list
    y_train = [ s_u_x/E, s_u_y/E, s_b_x/E, s_b_y/E, s_l_x/E, s_l_y/E, s_r_x/E, s_r_y/E ]
    
    
    return ns, ns_u, ns_l, x_train, y_train, E, mu, dx, h
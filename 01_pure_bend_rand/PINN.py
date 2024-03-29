"""

    This function is to initialize a PINN.
        
"""
import tensorflow as tf
from Dif_op_u import Dif_u
from Dif_op_v import Dif_v

def PINN(net1, net2, mu):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'net1'      [keras model]           \           : The trained FNN for displacment u;
        'net2'      [keras model]           \           : The trained FNN for displacment v;
        'mu'        [float]                 1           : Poisson ratio.
        
    Variables:
        Name        Type                    Size        Info.
        
        [xy]        [Array of float32]      ns*2        : Coordinates of all the sample points;
        [xy_u]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the top boundary of the beam;
        [xy_b]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the bottom boundary of the beam;
        [xy_l]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the left boundary of the beam;
        [xy_r]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the right boundary of the beam;
        [xy_f]      [Array of float32]      1*2         : Coordinates of the fixed sample point;
        [Gex]       [Array of float32]      ns*1        : Equilibrium equation for x direction of internal computational domain;
        [Gey]       [Array of float32]      ns*1        : Equilibrium equation for y direction of internal computational domain;
        [Gex_u]     [Array of float32]      ns_u*1      : Equilibrium equation for x direction on the top boundary of the beam;
        [Gey_u]     [Array of float32]      ns_u*1      : Equilibrium equation for y direction on the top boundary of the beam;
        [Gex_b]     [Array of float32]      ns_u*1      : Equilibrium equation for x direction on the bottom boundary of the beam;
        [Gey_b]     [Array of float32]      ns_u*1      : Equilibrium equation for y direction on the bottom boundary of the beam;
        [Gex_l]     [Array of float32]      ns_l*1      : Equilibrium equation for x direction on the left boundary of the beam;
        [Gey_l]     [Array of float32]      ns_l*1      : Equilibrium equation for y direction on the left boundary of the beam;
        [Gex_r]     [Array of float32]      ns_l*1      : Equilibrium equation for x direction on the right boundary of the beam;
        [Gey_r]     [Array of float32]      ns_l*1      : Equilibrium equation for y direction on the right boundary of the beam;
        [s_u_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the top boundary of the beam;
        [s_u_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the top boundary of the beam;
        [s_b_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the bottom boundary of the beam;
        [s_b_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the bottom boundary of the beam;
        [s_l_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the left boundary of the beam;
        [s_l_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the left boundary of the beam;
        [s_r_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the right boundary of the beam;
        [s_r_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the right boundary of the beam;
        [mu]        [float]                 1           : Poisson ratio.

    =================================================================================================================================    
    """

    ### Declare inputs
    xy = tf.keras.layers.Input(shape=(2,))
    xy_u = tf.keras.layers.Input(shape=(2,))
    xy_b = tf.keras.layers.Input(shape=(2,))
    xy_l = tf.keras.layers.Input(shape=(2,))
    xy_r = tf.keras.layers.Input(shape=(2,))
    
    ### Initialize the differential operators
    Dif1 = Dif_u(net1)
    Dif2 = Dif_v(net2)
    t = net2(xy_r)
    
    ### Obtain partial derivatives with respect to x and y
    U_x, U_y, U_xx, U_xy, U_yy = Dif1(xy)
    V_x, V_y, V_xx, V_xy, V_yy = Dif2(xy)
    U_u_x, U_u_y, U_u_xx, U_u_xy, U_u_yy = Dif1(xy_u)
    V_u_x, V_u_y, V_u_xx, V_u_xy, V_u_yy = Dif2(xy_u)
    U_b_x, U_b_y, U_b_xx, U_b_xy, U_b_yy = Dif1(xy_b)
    V_b_x, V_b_y, V_b_xx, V_b_xy, V_b_yy = Dif2(xy_b)
    U_l_x, U_l_y, U_l_xx, U_l_xy, U_l_yy = Dif1(xy_l)
    V_l_x, V_l_y, V_l_xx, V_l_xy, V_l_yy = Dif2(xy_l) 
    U_r_x, U_r_y, U_r_xx, U_r_xy, U_r_yy = Dif1(xy_r)
    V_r_x, V_r_y, V_r_xx, V_r_xy, V_r_yy = Dif2(xy_r) 
    
    ### Obtain the Lame constants for plain strain (or stress) problem
    ### plain strain
    la = mu/(1 + mu) / (1 - 2 * mu)
    nu = 1 / (1 + mu) / 2 
    
    #### plain stress
    # la = mu/(1 + mu) / (1 - mu)
    # nu = (1 + mu) / 2
    
    ### Obtain the residuals from stress boundary conditions
    s_u_x = nu * U_u_y + nu * V_u_x
    s_u_y = (2 * nu + la) * V_u_y + la * U_u_x
    s_b_x = nu * U_b_y + nu * V_b_x
    s_b_y = (2 * nu + la) * V_b_y + la * U_b_x
    s_l_y = nu * U_l_y + nu * V_l_x
    s_r_y = nu * U_r_y + nu * V_r_x
    s_l_x = (2 * nu + la) * U_l_x + la * V_l_y
    s_r_x = (2 * nu + la) * U_r_x + la * V_r_y
    
    ### Obtain the residuals from equilibrium equation
    Gex = (2 * nu + la) * U_xx + nu * U_yy + (nu + la) * V_xy
    Gey = (nu + la) * U_xy + (2 * nu + la) * V_yy + nu * V_xx
    
    return tf.keras.models.Model(
        inputs = [xy, xy_u, xy_b, xy_l, xy_r], \
            outputs = [Gex, Gey, s_u_x, s_u_y, s_b_x, s_b_y, s_l_x, s_l_y, s_r_x, s_r_y, t])
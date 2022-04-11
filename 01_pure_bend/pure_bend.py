"""

    This code is for the pure bending beam problem in "A Noval Way to Formulate the Loss Function 
    in Physics-Informed Neural Network for Solid Mechanics". DOI: 
    
    A pure bending beam is modeled here. The length and height of the beam are L = 1 m and H = 0.1 m. 
    The beam is bent by a moment M = 1/12 N∙m subjected on both sides of the beam. Due to the 
    geometric symmetry, only half of the beam is considered. To implement the moment M, we apply 
    a linear distributed force on the side of the beam, F(y) = 1000y N/m^2. The sample points are 
    uniformly distributed in the computational domain with 0.01 m intervals in all directions. We
    use two FNNs with the same structures to respectively predict the displacement field u and v.
    The Young's module E = 1000 Pa and the Poisson ratio is 0.3.
    
    This code is developed by @Jinshuai Bai and @Yuantong Gu. For more details, please contact: 
    jinshuai.bai@hdr.qut.edu.au
    yuantong.gu@qut.edu.au.
        
"""
#%%
"""

    Import library
    Including:
        
        Name             Source
        
        'NumPy'          https://numpy.org/
        'TensorFlow'     https://www.tensorflow.org/
        'SciPy'          https://scipy.org/
        'Matplotlib'     https://matplotlib.org/
        'time'           In Python3
        'os'             In Python3
        'Input_info'     Self developed code, found in this package
        'FNN'            Self developed code, found in this package
        'PINN'           Self developed code, found in this package
        'Opt'            Self developed code, found in this package
        'Visualization'  Self developed code, found in this package
        
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy.io

from Input_info import Input
from FNN import Net
from PINN import PINN
from Opt import Opt_lbfgsb
from Visualization import Vis

#%%
if __name__ == '__main__':
    """
    
        Input information for calculations.
        
    """
    
    ns, ns_u, ns_l, x_train, y_train, E, mu, dx, h = Input()
    
    """
    
        Build two Feedforward Neural Networks (FNN) to respectively predict the displacment u and v.
        
    """
    
    net_u = Net(n_input = 2, n_output = 1, layers=[10, 10, 10])
    net_v = Net(n_input = 2, n_output = 1, layers=[10, 10, 10])
    
    """
    
        Build a Physics-Informed Neural Networks (PINN) by the pre-built two neural networks.
        
    """
    
    pinn = PINN(net_u, net_v, mu)
    
    """
    
        Initialize the L-BFGS-B optimizer.
        
    """

    l_bfgs_b = Opt_lbfgsb(pinn, x_train, y_train, dx, h)
    
    """
    
        Train the PINN through the L-BFGS-B optimizer. Print the training time, final loss, and overall
        iterations for convergence. 
        
    """
    
    time_start = time.time()
    result = l_bfgs_b.fit()
    time_end = time.time()
    
    T = time_end-time_start
    L = result[1]
    it = result[2]['funcalls']
    print('-------------------------------------------------\n')
    print('Time cost is', T, 's')
    print('Final loss is', L, '')
    print('Training converges by', it, 'iterations\n')
    print('-------------------------------------------------\n')
    
    """
    
         Visualize the results
        
    """

    Vis(E, pinn, net_u, net_v, x_train[0], T, L, it)
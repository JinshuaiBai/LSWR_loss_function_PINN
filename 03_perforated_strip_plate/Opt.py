"""

    This class is to setup the Optimizer for neural networks training. Here we adopt L-BFGS-B optimizer
    provided by the SciPy package. Details of this optimizer can be found in: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    
"""
import scipy.optimize
import numpy as np
import tensorflow as tf

class Opt_lbfgsb:

    def __init__(self, pinn, x_train, y_train, dx, h, factr=10, pgtol=1e-10, m=50, maxls=50, maxiter=20000):
        """
        =================================================================================================================================
    
        Options:
            Name        Type                    Size        Info.
            
            'pinn'      [keras model]           \           : The PINN we want to train;
            'x_train'   [list]                  6           : PINN input list, contains all the coordinates information;
            'y_train'   [list]                  8           : PINN boundary condition list, contains all the force boundary conditions;
            'dx'        [float]                 1           : Sample points interval;
            'h'         [float]                 1           : ;
            'factr'     [int]                   1           : The optimizer option. Please refer to SciPy;
            'pgtol'     [float]                 1           : The optimizer option. Please refer to SciPy;
            'm'         [int]                   1           : The optimizer option. Please refer to SciPy;
            'maxls'     [int]                   1           : The optimizer option. Please refer to SciPy;
            'maxiter'   [int]                   1           : Maximum number of iterations for training.
    
        =================================================================================================================================    
        """
        # set options
        self.pinn = pinn
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.dx = dx
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.h = h

    def set_weights(self, flat_weights):
        """
        
            Set weights to the model.
        
        """
        # get model weights
        shapes = [ w.shape for w in self.pinn.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.pinn.set_weights(weights)

    @tf.function
    def Loss_grad(self, x, y):
        """
        
            Formulate the loss function. Output the grad of loss with respect to all trainable variables.
            
        """
        nx=-x[5][...,0,tf.newaxis]/0.2
        ny=-x[5][...,1,tf.newaxis]/0.2
        with tf.GradientTape() as g:
            l1 = self.h * self.dx ** 2 * (tf.reduce_sum(tf.square(self.pinn(x)[0])) + \
                tf.reduce_sum(tf.square(self.pinn(x)[1])))
            l2 = (self.dx * (tf.reduce_sum(tf.square(self.pinn(x)[2]-y[0])) + \
                tf.reduce_sum(tf.square(self.pinn(x)[3]-y[1]))) + \
                self.dx * (tf.reduce_sum(tf.square(self.pinn(x)[4]-y[2]))) + \
                self.dx * (tf.reduce_sum(tf.square(self.pinn(x)[7]-y[5]))) + \
                self.dx * (tf.reduce_sum(tf.square(self.pinn(x)[8]-y[6])) + \
                tf.reduce_sum(tf.square(self.pinn(x)[9]-y[7]))))
            l3 = (self.dx * (tf.reduce_sum(tf.square(self.pinn(x)[10] * nx + self.pinn(x)[12] * ny)) + \
                tf.reduce_sum(tf.square(self.pinn(x)[12] * nx + self.pinn(x)[11] * ny))))
            loss = l1 + l2 + l3
        grads = g.gradient(loss, self.pinn.trainable_variables)      
        return loss, grads, l1, l2, l3

    def Loss(self, weights):
        """
        
            Write down losses in L.txt file. Visualize the losses in the current iteration.
        
        """
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads, l1, l2, l3 = self.Loss_grad(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        print('L1 =',l1.numpy(),'   L2 =',l2.numpy(),'   L3 =',l3.numpy(),'')
        # with open('L.txt','a') as f:
        #     f.write(str(l1.numpy())+' '+str(l2.numpy())+' '+str(l3.numpy())+'\n')
        loss = loss.numpy().astype('float64')
        # print(grads)
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def fit(self):
        """
        
            Train the PINN by using the L-BFGS-B algorithm.
        
        """
        # get initial weights as a flat vector
        initial_weights = np.concatenate([ w.flatten() for w in self.pinn.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (Provided by Scipy package)')
        print('Initializing the framework ...')
        result = scipy.optimize.fmin_l_bfgs_b(func=self.Loss, x0=initial_weights,
            factr=self.factr, pgtol=self.pgtol, m=self.m, maxls=self.maxls, maxiter=self.maxiter)
        return result
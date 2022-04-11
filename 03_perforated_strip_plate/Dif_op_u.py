"""

    This calss is the costumize keras layer to calculate derivatives for displacement u. The output 
    derivatives with respect to the input of the FNN can be obtained through the GradientTape function
    provided by TensorFlow.

"""
import tensorflow as tf

class Dif_u(tf.keras.layers.Layer):

    def __init__(self, fnn, **kwargs):
        """
        =================================================================================================================================
        
        Options:
            Name        Type                    Size        Info.
            
            'fnn'       [keras model]           \           : The Feedforward Neural Network.
        
        =================================================================================================================================    
        """
        self.fnn = fnn
        super().__init__(**kwargs)
    
    def call(self, xy):
        """
        
            Calculate the derivatives through the GradientTape function.
            
        """
        x, y = (xy[..., i, tf.newaxis] for i in range(xy.shape[-1]))
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)
                temp = self.fnn(tf.concat([x, y], axis=-1))
                U = temp * x
            U_x = g.gradient(U, x)
            U_y = g.gradient(U, y)
            del g
        U_xx = gg.gradient(U_x, x)
        U_xy = gg.gradient(U_x, y)
        U_yy = gg.gradient(U_y, y)
        del gg
        
        return U_x, U_y, U_xx, U_xy, U_yy

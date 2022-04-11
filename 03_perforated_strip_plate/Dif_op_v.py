"""

    This calss is the costumize keras layer to calculate derivatives for displacement v. The output 
    derivatives with respect to the input of the FNN can be obtained through the GradientTape function
    provided by TensorFlow.
        
"""
import tensorflow as tf

class Dif_v(tf.keras.layers.Layer):

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
                V = temp * y
            V_x = g.gradient(V, x)
            V_y = g.gradient(V, y)
            del g
        V_xx = gg.gradient(V_x, x)
        V_xy = gg.gradient(V_x, y)
        V_yy = gg.gradient(V_y, y)
        del gg

        return V_x, V_y, V_xx, V_xy, V_yy

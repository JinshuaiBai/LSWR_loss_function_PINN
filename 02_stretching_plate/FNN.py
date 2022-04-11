"""

    This function is to set up a FNN. The FNN is built up with the Tensorflow. 
        
"""
import tensorflow as tf

def swish(self, x):
        """
        Swish activation function.

        Args:
            x: activation input.

        Returns:
            Swish output.
        """
        return x * tf.math.sigmoid(x)

def Net(n_input = 2, n_output = 1, layers = [20,20], acti_fun = 'tanh', k_init = 'he_normal'):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'n_input'   [int]                   1           : Number of inputs for the FNN;
        'n_output'  [int]                   1           : Number of outputs for the FNN;
        'layers'    [list]                  costmize    : Size of the FNN;
        'acti_fun'  [str]                   \           : The activation function used after each layer;
        'k_init'    [str]                   \           : The kernal initializaiton method.
        
    Variables:
        Name        Type                    Size        Info.
        
        [x]         [keras input]           \           : Input of the neural network;
        [y]         [keras output]          \           : Output of the neural network;

    =================================================================================================================================    
    """
    x = tf.keras.layers.Input(shape=(n_input))
    temp = x
    for l in layers:
        temp = tf.keras.layers.Dense(l, activation=acti_fun, kernel_initializer=k_init)(temp)
    y = tf.keras.layers.Dense(n_output, kernel_initializer=k_init)(temp)

    return tf.keras.models.Model(inputs = x, outputs = y)

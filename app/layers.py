import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, validation_embedding=None):
        # Debugging print to understand input types
        print("L1Dist Layer Inputs:")
        print(f"inputs type: {type(inputs)}")
        print(f"validation_embedding type: {type(validation_embedding)}")
        print(f"validation_embedding value: {validation_embedding}")
        
        # Handle different input scenarios
        if validation_embedding is None or isinstance(validation_embedding, (list, str)):
            # If inputs is a list with two elements
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                input_embedding, validation_embedding = inputs
            else:
                raise ValueError("L1Dist layer requires two inputs: input_embedding and validation_embedding")
        else:
            input_embedding = inputs
        
        # Ensure inputs are float32 tensors
        try:
            # Convert input_embedding
            if not isinstance(input_embedding, tf.Tensor):
                input_embedding = tf.convert_to_tensor(input_embedding, dtype=tf.float32)
            
            # Handle validation_embedding carefully
            if isinstance(validation_embedding, (list, np.ndarray, str)):
                # If it's a string or list, raise an error
                if isinstance(validation_embedding, (str, list)):
                    raise ValueError(f"Invalid validation_embedding type: {type(validation_embedding)}")
                
                # Convert to tensor
                validation_embedding = tf.convert_to_tensor(validation_embedding, dtype=tf.float32)
            
            # Compute L1 distance (absolute difference)
            return tf.math.abs(input_embedding - validation_embedding)
        
        except Exception as e:
            print(f"Error in L1Dist layer: {e}")
            raise
    
    def compute_output_shape(self, input_shape):
        # Return the same shape as the input
        return input_shape
    
    def get_config(self):
        # Needed for model saving/loading
        base_config = super().get_config()
        return base_config
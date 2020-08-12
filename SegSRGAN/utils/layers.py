"""
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

from tensorflow.python.ops import array_ops
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np


class activation_SegSRGAN(Layer):
    def __init__(self, int_channel=0, seg_channel=1, activation='sigmoid', is_residual=True,nb_classe_mask=2,fit_mask=False, **kwargs):
        self.seg_channel = seg_channel
        self.int_channel = int_channel
        self.activation = activation
        self.is_residual = is_residual
        self.nb_classe_mask = nb_classe_mask
        self.fit_mask = fit_mask
        super(activation_SegSRGAN, self).__init__(**kwargs)
        
        # For now the activation function will apply soigmoid on channel seg_channel: and residual activation on int_channel. The residual part must be modify on int_channel: if we want to consider more input image 

    def build(self, input_shapes):
        super(activation_SegSRGAN, self).build(input_shapes)

    def call(self, inputs):
        recent_input = inputs[0] # pred
        first_input = inputs[1] # im
        
        if self.activation == 'sigmoid':
            segmentation_label = K.sigmoid(recent_input[:, self.seg_channel, :, :, :]) #return a array enven if self.segchannel is the last indice
            segmentation = K.expand_dims(segmentation_label, axis=1)
            
            if self.fit_mask : 
                segmentation_mask = K.softmax(recent_input[:, 2:(2+self.nb_classe_mask), :, :, :],axis=-4) 
            
                segmentation = K.concatenate([segmentation,segmentation_mask], axis=1)
        else:
            assert'Do not support'
        intensity = recent_input[:, self.int_channel, :, :, :]
        
        # Adding channel
        intensity = K.expand_dims(intensity, axis=1)
        
        if self.is_residual :
            # residual
            residual_intensity = first_input - intensity
            # correction de l'image interpolee
            print("A residual model has been initialized")
        else:
            residual_intensity = intensity
            print("A non residual model has been initialized")

        return K.concatenate([residual_intensity,segmentation], axis=1)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    


class InstanceNormalization3D(Layer):
    ''' Thanks for github.com/jayanthkoushik/neural-style 
    and https://github.com/PiscesDream/CycleGAN-keras/blob/master/CycleGAN/layers/normalization.py'''
    def __init__(self, **kwargs):
        super(InstanceNormalization3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[1],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[1],), initializer="zero", trainable=True)
        super(InstanceNormalization3D, self).build(input_shape)

    def call(self, x):
        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(K.expand_dims(tensor, -1), -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        hwk = K.cast(x.shape[2] * x.shape[3] * x.shape[4], K.floatx())
        mu = K.sum(x, [-1, -2, -3]) / hwk
        mu_vec = image_expand(mu) 
        sig2 = K.sum(K.square(x - mu_vec), [-1, -2, -3]) / hwk
        y = (x - mu_vec) / (K.sqrt(image_expand(sig2)) + K.epsilon())

        scale = batch_image_expand(self.scale)
        shift = batch_image_expand(self.shift)
        return scale*y + shift 

    def compute_output_shape(self, input_shape):
        return input_shape



class ReflectPadding3D(Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectPadding3D, self).__init__(**kwargs)
        self.padding = ((padding, padding), (padding, padding), (padding, padding))

    def compute_output_shape(self, input_shape):
        if input_shape[2] is not None:
            dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
        else:
            dim1 = None
        if input_shape[3] is not None:
            dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
        else:
            dim2 = None
        if input_shape[4] is not None:
            dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
        else:
            dim3 = None
        return (input_shape[0],
                input_shape[1],
                dim1,
                dim2,
                dim3)

    def call(self, inputs):
        pattern = [[0, 0], [0, 0], 
                   [self.padding[0][0], self.padding[0][1]],
                   [self.padding[1][0], self.padding[1][1]], 
                   [self.padding[2][0], self.padding[2][1]]]
            
        return array_ops.pad(inputs, pattern, mode= "REFLECT")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, Conv3DTranspose
from keras.layers import UpSampling3D, Dropout, BatchNormalization

'''
Orignal code publication : https://github.com/pietz/unet-keras/blob/master/unet.py.

adapatation 3D of U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, acti, bn, res, do=0): #dim = number of features, 
	n = Conv3D(dim, 3, activation=acti, padding='same',data_format='channels_first')(m)
	n = BatchNormalization(axis=1)(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv3D(dim, 3, activation=acti, padding='same',data_format='channels_first')(n)
	n = BatchNormalization(axis=1)(n) if bn else n
	return Concatenate(axis=1)([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling3D(data_format='channels_first')(n) if mp else Conv3D(dim, 3, strides=2, padding='same',data_format='channels_first')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling3D(size=(2, 2, 2),data_format='channels_first')(m)
			m = Conv3D(dim, 2, activation=acti, padding='same',data_format='channels_first')(m)
		else:
			m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same',data_format='channels_first')(m)
		n = Concatenate(axis=1)([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv3D(out_ch, 1, activation='sigmoid',data_format='channels_first')(o)
	return Model(inputs=[i], outputs=[o])

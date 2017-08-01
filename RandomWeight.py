#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 26

The goal of this code is to draw random weight for the VGG 19 architecture

@author: nicolas
"""

import scipy
import numpy as np
import Style_Transfer as st
import tensorflow as tf
import os

VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)

VGG19_LAYERS_INDICES = {'conv1_1' : 0,'conv1_2' : 2,'conv2_1' : 5,'conv2_2' : 7,
	'conv3_1' : 10,'conv3_2' : 12,'conv3_3' : 14,'conv3_4' : 16,'conv4_1' : 19,
	'conv4_2' : 21,'conv4_3' : 23,'conv4_4' : 25,'conv5_1' : 28,'conv5_2' : 30,
	'conv5_3' : 32,'conv5_4' : 34}

path_origin = '/home/nicolas/Images/original/'

def draw_random_weight(filename='random_net.mat',sigma=0.0015):
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	for name in VGG19_LAYERS_INDICES.keys():
		index_in_vgg = VGG19_LAYERS_INDICES[name]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		kernels_random = np.random.normal(0.0, sigma, size=kernels.shape).astype('float32')
		bias_random = np.random.normal(0.0, sigma, size=bias.shape).astype('float32')
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels_random
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias_random
	scipy.io.savemat(filename,vgg_rawnet)
	
def draw_random_weight_fromkernels(filename='random_net.mat'):
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	for name in VGG19_LAYERS_INDICES.keys():
		index_in_vgg = VGG19_LAYERS_INDICES[name]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		std_kernels = np.std(kernels)
		std_bias = np.std(bias)
		kernels_random = np.random.normal(0.0, std_kernels, size=kernels.shape).astype('float32')
		bias_random = np.random.normal(0.0, std_bias, size=bias.shape).astype('float32')
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels_random
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias_random
	scipy.io.savemat(filename,vgg_rawnet)
		
def draw_random_weight_unif(filename='random_net.mat',sigma=0.015):
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	for name in VGG19_LAYERS_INDICES.keys():
		index_in_vgg = VGG19_LAYERS_INDICES[name]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		width, height, in_channels, out_channels = kernels.shape
		u = 1./np.sqrt(width*height*in_channels)
		kernels_random = np.random.uniform(-u, u, size=kernels.shape).astype('float32')
		bias_random = np.random.uniform(-u, u, size=bias.shape).astype('float32')
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels_random
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias_random
	scipy.io.savemat(filename,vgg_rawnet)
	
def get_list_of_images():
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)		

def normalization_of_weight(filename='random_net.mat'):
	
	VGG19_mat=filename
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	input_image = np.zeros((1,256,256,3)).astype('float32')
	net = st.net_preloaded(vgg_layers, input_image,pooling_type='avg',padding='SAME')
	
	placeholder = tf.placeholder(tf.float32, shape=input_image.shape)
	assign_op = net['input'].assign(placeholder)
	sess = sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	dirs = get_list_of_images()
	dirs_result = {}
	for layer in VGG19_LAYERS:
		print(layer)
		tmp = None
		for img_path in dirs:
			img = scipy.misc.imread(path_origin+img_path, mode="RGB")
			img_reshape = scipy.misc.imresize(img, (256,256), interp='bilinear').astype('float32')
			img_reshape = st.preprocess(img_reshape)
			sess.run(assign_op, {placeholder: img_reshape})
			f_layer = sess.run(net[layer])
			if(tmp is None):
				tmp = f_layer
			else:
				tmp += f_layer
		tmp /= len(dirs)
		dirs_result[layer] = np.mean(tmp,axis=(0,1,2))
	
	old_norm = [1]*3
	print(VGG19_LAYERS_INDICES.keys())
	VGG19_layer_groupe = (('conv1_1', 'relu1_1'), ('conv1_2', 'relu1_2'),
	('conv2_1', 'relu2_1'), ('conv2_2', 'relu2_2'), ('conv3_1', 'relu3_1'),('conv3_2', 'relu3_2'), ('conv3_3','relu3_3'),('conv3_4', 'relu3_4'),
	('conv4_1', 'relu4_1'), ('conv4_2', 'relu4_2'), ('conv4_3',	'relu4_3'), ('conv4_4', 'relu4_4'),('conv5_1', 'relu5_1'), ('conv5_2', 'relu5_2'), ('conv5_3','relu5_3'), ('conv5_4', 'relu5_4'))
	for (l1,l2) in VGG19_layer_groupe:
		print(l1,l2)
		mean = dirs_result[l2]
		
		index_in_vgg = VGG19_LAYERS_INDICES[l1]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		width, height, in_channels, out_channels = kernels.shape
		old_norm[old_norm==0] = 1
		mean[mean==0]=1
		print(mean)
		for i in range(in_channels):
			kernels[:,:,i,:] *= old_norm[i]
			#bias[i] = mean[i]
		for i in range(out_channels):
			kernels[:,:,:,i] /= mean[i]
			bias[i] /= mean[i]
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias
		old_norm = mean
		
	scipy.io.savemat('random_net.mat',vgg_rawnet)
	
	
		
#To stabilize the reconstruction quality, we apply a greedy approach to build a “stacked” random
#weight network ranVGG based on the VGG-19 architecture. Select one single image as the reference
#image and starting from the first convolutional layer, we build the stacked random weight VGG by
#sampling, selecting and fixing the weights of each layer in forward order. For the current layer l,
#fix the weights of the previous l − 1 layers and sample several sets of random weights connecting
#the l th layer. Then reconstruct the target image using the rectified representation of layer l, and
#choose weights yielding the smallest loss. Experiments in the next section show our success on the
#reconstruction by using the untrained, random weight CNN, ranVGG.


def draw_random_weightGlorot(filename='random_net.mat'):
	"""
	Inspired from https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
	"""
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	for name in VGG19_LAYERS_INDICES.keys():
		index_in_vgg = VGG19_LAYERS_INDICES[name]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		shape = kernels.shape

		gain = np.sqrt(2)
		n1, n2 = shape[:2]
		receptive_field_size = np.prod(shape[2:])
		print(n1, n2,receptive_field_size)
		std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
		kernels_random = np.random.normal(0.0, std, size=kernels.shape).astype('float32')
		bias_random = np.random.normal(0.0, std, size=bias.shape).astype('float32')
		
		#u = 1./np.sqrt(np.prod(shape[0:3]))
		#kernels_random = np.random.uniform(-u, u, size=kernels.shape).astype('float32')
		#bias_random = np.random.uniform(-u, u, size=bias.shape).astype('float32')
		
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels_random
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias_random
	scipy.io.savemat(filename,vgg_rawnet)       
	

def draw_random_weightHE(filename='random_net.mat'):
	"""
	Inspired from https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
	"""
	VGG19_mat='imagenet-vgg-verydeep-19.mat'
	vgg_rawnet = scipy.io.loadmat(VGG19_mat)
	vgg_layers = vgg_rawnet['layers'][0]
	for name in VGG19_LAYERS_INDICES.keys():
		index_in_vgg = VGG19_LAYERS_INDICES[name]
		kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
		bias = vgg_layers[index_in_vgg][0][0][2][0][1]
		width, height, in_channels, out_channels =  kernels.shape
		kernels_random = (np.random.randn(width, height, in_channels, out_channels) * np.sqrt(2.0/in_channels)).astype('float32')
		bias_random = np.zeros(bias.shape).astype('float32')
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][0] = kernels_random
		vgg_rawnet['layers'][0][index_in_vgg][0][0][2][0][1] = bias_random
	scipy.io.savemat(filename,vgg_rawnet)
	

		
if __name__ == '__main__':
	draw_random_weight()   
	#draw_random_weightGlorot()
	#normalization_of_weight()
	#draw_random_weight_unif()
	#draw_random_weight_fromkernels()
	#draw_random_weightHE()


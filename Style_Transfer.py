#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:32:49 2017

The goal of this script is to code the Style Transfer Algorithm 

Inspired from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
and https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb

@author: nicolas
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0' # 1 to remove info, 2 to remove warning and 3 for all
import tensorflow as tf
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import math
from tensorflow.python.client import timeline
from Arg_Parser import get_parser_args 
import utils
from numpy.fft import fft2, ifft2
from skimage.color import gray2rgb

# Name of the 19 first layers of the VGG19
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

# Max et min value from the ImageNet databse mean
clip_value_min=-124
clip_value_max=152


# TODO change that it is not a really good idea to have globla variable in Python
content_layers = [('conv4_2',1.)]
#style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
#style_layers = [('conv1_1',1.)]
# TODO : be able to choose more quickly the different parameters

# TODO segment the vgg loader, and the rest

def plot_image(path_to_image):
	"""
	Function to plot an image
	"""
	img = Image.open(path_to_image)
	plt.imshow(img)
	
def get_vgg_layers():
	"""
	Load the VGG 19 layers
	"""
	VGG19_mat ='imagenet-vgg-verydeep-19.mat' 
	# The vgg19 network from http://www.vlfeat.org/matconvnet/pretrained/
	try:
		vgg_rawnet = scipy.io.loadmat(VGG19_mat)
		vgg_layers = vgg_rawnet['layers'][0]
	except(FileNotFoundError):
		print("The path to the VGG19_mat is not right or the .mat is not here")
		raise
	return(vgg_layers)

def net_preloaded(vgg_layers, input_image,pooling_type='avg',padding='SAME'):
	"""
	This function read the vgg layers and create the net architecture
	We need the input image to know the dimension of the input layer of the net
	"""
	net = {}
	_,height, width, numberChannels = input_image.shape # In order to have the right shape of the input
	current = tf.Variable(np.zeros((1, height, width, numberChannels), dtype=np.float32))
	net['input'] = current
	for i, name in enumerate(VGG19_LAYERS):
		kind = name[:4]
		if(kind == 'conv'):
			# Only way to get the weight of the kernel of convolution
			# Inspired by http://programtalk.com/vs2/python/2964/facenet/tmp/vggverydeep19.py/
			kernels = vgg_layers[i][0][0][2][0][0]
			bias = vgg_layers[i][0][0][2][0][1]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = tf.constant(np.transpose(kernels, (1, 0, 2, 3)))
			# TODO check if the convolution is in the right order
			bias = tf.constant(bias.reshape(-1))
			current = _conv_layer(current, kernels, bias,name,padding) 
			# Update the  variable named current to have the right size
		elif(kind == 'relu'):
			current = tf.nn.relu(current,name=name)
		elif(kind == 'pool'):
			current = _pool_layer(current,name,pooling_type,padding)
		net[name] = current
		#print(name,current.shape)

	assert len(net) == len(VGG19_LAYERS) +1 # Test if the length is right 
	return(net)

def _conv_layer(input, weights, bias,name,padding='SAME'):
	"""
	This function create a conv2d with the already known weight and bias
	
	conv2d :
	Computes a 2-D convolution given 4-D input and filter tensors.
	input: A Tensor. Must be one of the following types: half, float32, float64
	Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
	a filter / kernel tensor of shape 
	[filter_height, filter_width, in_channels, out_channels]
	"""
	if(padding=='SAME'):
		conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
			padding=padding,name=name)
	elif(padding=='VALID'):
		input = get_img_2pixels_more(input)
		conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
			padding='VALID',name=name)
	# We need to impose the weights as constant in order to avoid their modification
	# when we will perform the optimization
	return(tf.nn.bias_add(conv, bias))

def get_img_2pixels_more(input):
	new_input = tf.concat([input,input[:,0:2,:,:]],axis=1)
	new_input = tf.concat([new_input,new_input[:,:,0:2,:]],axis=2)
	return(new_input)

def _pool_layer(input,name,pooling_type='avg',padding='SAME'):
	"""
	Average pooling on windows 2*2 with stride of 2
	input is a 4D Tensor of shape [batch, height, width, channels]
	Each pooling op uses rectangular windows of size ksize separated by offset 
	strides in the avg_pool function 
	"""
	if(padding== 'VALID'): # Test if paire ou impaire !!! 
		_,h,w,_ = input.shape
		if not(h%2==0):
			input = tf.concat([input,input[:,0:2,:,:]],axis=1)
		if not(w%2==0):
			input = tf.concat([input,input[:,:,0:2,:]],axis=2)
	if pooling_type == 'avg':
		pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding=padding,name=name) 
	elif pooling_type == 'max':
		pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding=padding,name=name) 
	return(pool)

def sum_content_losses(sess, net, dict_features_repr,M_dict):
	"""
	Compute the content term of the loss function
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of the content image representation thanks to the net
	"""
	length_content_layers = float(len(content_layers))
	weight_help_convergence = 10**10 # Need to multiply by 120000 ?
	content_loss = 0
	for layer, weight in content_layers:
		M = M_dict[layer[:5]]
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		content_loss +=  tf.nn.l2_loss(tf.subtract(P,F))*(
			weight*weight_help_convergence/(length_content_layers*(tf.to_float(M)**2)))
	return(content_loss)

def sum_style_losses(sess, net, dict_gram,M_dict):
	"""
	Compute the style term of the loss function 
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of Gram Matrices
	- the dictionnary of the size of the image content through the net
	"""
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	# Info for the vgg19
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 10**(9) # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		A = dict_gram[layer]
		A = tf.constant(A)
		# Get the value of this layer with the generated image
		M = M_dict[layer[:5]]
		x = net[layer]
		G = gram_matrix(x,N,M)
		style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
		# TODO selon le type de style voulu soit reshape the style image sinon Mcontenu/Mstyle
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	return(total_style_loss)

def compute_4_moments(x):
	"""
	Compute the 4 first moments of the features (response of the kernel) 
	of a 4D Tensor
	"""
	# TODO : this is biased moment !! 
	mean_x = tf.reduce_mean(x, axis=[0,1,2])
	variance_x = tf.subtract(tf.reduce_mean(tf.pow(x,2), axis=[0,1,2]),mean_x)
	sig_x = tf.sqrt(variance_x)
	skewness_x = tf.reduce_mean(tf.pow(tf.divide(tf.subtract(x,mean_x),sig_x),3), axis=[0,1,2])
	kurtosis_x = tf.reduce_mean(tf.pow(tf.divide(tf.subtract(x,mean_x),sig_x),4), axis=[0,1,2])
	return(mean_x,variance_x,skewness_x,kurtosis_x)
		
		
def compute_n_moments(x,n,axis=[0,1,2]):
	"""
	Compute the n first moments of the features (response of the kernel)
	"""
	assert(n > 0)
	mean_x = tf.reduce_mean(x,axis=axis)
	list_of_moments = [mean_x]
	if(n>1):
		variance_x = tf.subtract(tf.reduce_mean(tf.pow(x,2), axis=axis),mean_x)
		list_of_moments += [variance_x]
		sig_x = tf.sqrt(variance_x)
	if(n>2):
		for r in range(3,n+1,1):
			moment_r = tf.reduce_mean(tf.pow(tf.divide(tf.subtract(x,mean_x),sig_x),r), axis=axis) # Centré/réduit
			# TODO : change that to some thing more optimal : pb computation of the power several times
			list_of_moments += [moment_r]
	return(list_of_moments)
	
def compute_Lp_norm(x,p):
	"""
	Compute the p first Lp norm of the features
	"""
	assert(p > 0)
	list_of_Lp = []
	for r in range(1,p+1,1):
		L_r_x = tf.pow(tf.reduce_mean(tf.pow(tf.abs(x),r), axis=[0,1,2]),1./r) 
		#F_x = tf.reshape(x,[M_i_1,N_i_1])
		#L_r_x =tf.norm(x,ord=r,axis=[0,1],name=str(r)) 
		# TODO : change that to some thing more optimal : pb computation of the power several times
		list_of_Lp += [L_r_x]
	return(list_of_Lp)
		
def sum_style_stats_loss(sess,net,image_style,M_dict):
	"""
	Compute a loss that is the l2 norm of the 4th moment of the optimization
	"""
	length_style_layers = float(len(style_layers))
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9 # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer] # response to the layer 
		mean_x,variance_x,skewness_x,kurtosis_x = compute_4_moments(x)
		mean_a,variance_a,skewness_a,kurtosis_a = compute_4_moments(a)
		style_loss = tf.nn.l2_loss(tf.subtract(mean_x,mean_a)) + tf.nn.l2_loss(tf.subtract(variance_x,variance_a)) + tf.nn.l2_loss(tf.subtract(skewness_x,skewness_a)) + tf.nn.l2_loss(tf.subtract(kurtosis_x,kurtosis_a))
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	return(total_style_loss)

def loss_n_moments(sess,net,image_style,M_dict,n):
	"""
	Compute a loss that is the l2 norm of the nth moment of the optimization
	"""
	length_style_layers = float(len(style_layers))
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9 # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer] # response to the layer 
		moments_x = compute_n_moments(x,n)
		moments_a = compute_n_moments(a,n)
		style_loss = sum(map(tf.nn.l2_loss,map(tf.subtract, moments_x,moments_a)))
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers) # Normalized by the number of features N
		total_style_loss += style_loss
	return(total_style_loss)

def loss_n_stats(sess,net,image_style,M_dict,n,TypeOfComputation='moments'):
	"""
	Compute a loss that is the l2 norm of the n element of a statistic on 
	the features maps : moments or norm
	"""
	length_style_layers = float(len(style_layers))
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9 # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer] # response to the layer 
		if(TypeOfComputation=='moments'):
			stats_x = compute_n_moments(x,n)
			stats_a = compute_n_moments(a,n)
		elif(TypeOfComputation=='Lp'):
			stats_x = compute_Lp_norm(x,n)
			stats_a = compute_Lp_norm(a,n)
		style_loss = sum(map(tf.nn.l2_loss,map(tf.subtract, stats_x,stats_a)))
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers) # Normalized by the number of features N
		total_style_loss += style_loss
	return(total_style_loss)

def loss_p_norm(sess,net,image_style,M_dict,p): # Faire une fonction génértique qui prend en entree le type de norme !!! 
	length_style_layers = float(len(style_layers))
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9 # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# For one layer
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer] # response to the layer 
		L_p_x = compute_Lp_norm(x,p) # Les Lp sont positives, on cherche juste à egaliser les énergies la
		L_p_a = compute_Lp_norm(a,p)
		style_loss = sum(map(tf.nn.l2_loss,map(tf.subtract, L_p_x,L_p_a)))
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers) # Normalized by the number of features N
		total_style_loss += style_loss
	return(total_style_loss)
	

def loss_crosscor_inter_scale(sess,net,image_style,M_dict,sampling='down',pooling_type='avg'):
	"""
	Compute a loss that is the l2 norm of the cross correlation of the previous band
	The sampling argument is down for downsampling and up for up sampling
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9 # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0.
	sess.run(net['input'].assign(image_style))
	if(length_style_layers_int > 1):
		for i in range(length_style_layers_int-1):
			layer_i, weight_i = style_layers[i]
			layer_i_1, weight_i_1 = style_layers[i+1]
			N_i = style_layers_size[layer_i[:5]]
			N_i_1 = style_layers_size[layer_i_1[:5]]
			M_i_1 = M_dict[layer_i_1[:5]]
			#print("M_i,M_i_1,N_i",M_i,M_i_1,N_i)
			x_i = net[layer_i]
			x_i_1 = net[layer_i_1]
			a_i = sess.run(net[layer_i])
			a_i_1 = sess.run(net[layer_i_1]) # TODO change this is suboptimal because youcompute twice a_i !! 
			if(sampling=='down'):
				if(pooling_type=='avg'):
					x_i = tf.nn.avg_pool(x_i, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
					a_i = tf.nn.avg_pool(a_i, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
				elif(pooling_type == 'max'):
					x_i = tf.nn.max_pool(x_i, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME') 
					a_i = tf.nn.max_pool(a_i, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME') 
				_,height,width,_ = x_i.shape
				M_i = tf.to_int32(height*width)
			elif(sampling=='up'):
				_,new_height,new_width,_ = x_i.shape
				_,height,_,_ = x_i_1.shape
				if(layer_i[:5]==layer_i_1[:5]):
					factor = 1 # Not upsample
				else:
					factor = 2
				upsample_filter_np = utils.bilinear_upsample_weights(factor,N_i_1)
				x_i_1 = tf.nn.conv2d_transpose(x_i_1, upsample_filter_np,
						output_shape=[1, tf.to_int32(new_height), tf.to_int32(new_width), N_i_1],
						strides=[1, factor, factor, 1])
				a_i_1 = tf.nn.conv2d_transpose(a_i_1, upsample_filter_np,
						output_shape=[1, tf.to_int32(new_height), tf.to_int32(new_width), N_i_1],
						strides=[1, factor, factor, 1])
				M_i = tf.to_int32(new_height*new_width)
				M_i_1 = M_i
			F_x_i = tf.reshape(x_i,[M_i,N_i])
			F_x_i_1 = tf.reshape(x_i_1,[M_i_1,N_i_1])
			G_x = tf.matmul(tf.transpose(F_x_i),F_x_i_1)
			G_x /= tf.to_float(M_i)
			F_a_i = tf.reshape(a_i,[M_i,N_i])
			F_a_i_1 = tf.reshape(a_i_1,[M_i_1,N_i_1])
			G_a = tf.matmul(tf.transpose(F_a_i),F_a_i_1)
			G_a /= tf.to_float(M_i)
			style_loss = tf.nn.l2_loss(tf.subtract(G_x,G_a))  # output = sum(t ** 2) / 2
			# TODO selon le type de style voulu soit reshape the style image sinon Mcontenu/Mstyle
			weight= (weight_i + weight_i_1) /2.
			style_loss *=  weight * weight_help_convergence  / (2.*(N_i*N_i_1)*length_style_layers)
			total_style_loss += style_loss
	return(total_style_loss)

def loss_autocorr(sess,net,image_style,M_dict):
	"""
	Computation of the autocorrelation of the filter 
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	weight_help_convergence = 10**9
	total_style_loss = 0.
	x_temp = {}
	sess.run(net['input'].assign(image_style))	
	for layer, weight in style_layers:
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		#R_a = (ifft2(fft2(a) * fft2(a).conj()).real)/M
		#R_x = x_temp[layer]
		x = net[layer]
		F_x = tf.fft2d(tf.complex(x,0.))
		#print(F_x.shape)
		R_x = tf.real(tf.multiply(F_x,tf.conj(F_x)))
		R_x /= tf.to_float(M)
		#print(R_x.shape)
		F_a = tf.fft2d(tf.complex(a,0.))
		R_a = tf.real(tf.multiply(F_a,tf.conj(F_a)))
		R_a /= tf.to_float(M)
		style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)
	
def loss_spectrum(sess,net,image_style,M_dict):
	"""
	Computation of the spectrum loss
	"""
	
	#x = net['input']
	#F_x = tf.fft2d(tf.complex(x,0.))
	#F_a = tf.fft2d(tf.complex(image_style,0.))
	#innerProd = tf.reduce_sum( tf.multiply(F_x,tf.conj(F_a)), 1, keep_dims=True )  # sum(ftIm .* conj(ftRef), 3);
	#dephase = innerProd ./ (abs(innerProd) + eps);
	#ftNew = bsxfun(@times, ftRef, dephase);
	#prod = tf.multiply(F_x,tf.conj(F_a))
	#prod /= 
	
	#weight_help_convergence = 10**9
	#[b, h, w, d] = x.get_shape()
	#b, h, w, d = tf.to_int32(b),tf.to_int32(h),tf.to_int32(w),tf.to_int32(d)
	#tv_y_size = tf.to_float(b * (h-1) * w * d)
	#tv_x_size = tf.to_float(b * h * (w-1) * d)
	#loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:]) 
	#loss_y /= tv_y_size
	#loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
	#loss_x /= tv_x_size
	#loss = 2 * weight_help_convergence * (loss_y + loss_x)
	#loss = tf.cast(loss, tf.float32)
	
	
	#length_style_layers_int = len(style_layers)
	#length_style_layers = float(length_style_layers_int)
	#style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	#weight_help_convergence = 10**9
	#total_style_loss = 0.
	#x_temp = {}
	#sess.run(net['input'].assign(image_style))	
	#for layer, weight in style_layers:
		#N = style_layers_size[layer[:5]]
		#M = M_dict[layer[:5]]
		#a = sess.run(net[layer])
		##R_a = (ifft2(fft2(a) * fft2(a).conj()).real)/M
		##R_x = x_temp[layer]
		#x = net[layer]
		#F_x = tf.fft2d(tf.complex(x,0.))
		#print(F_x.shape)
		#R_x = tf.real(tf.multiply(F_x,tf.conj(F_x)))
		#R_x /= tf.to_float(M)
		#print(R_x.shape)
		#F_a = tf.fft2d(tf.complex(a,0.))
		#R_a = tf.real(tf.multiply(F_a,tf.conj(F_a)))
		#R_a /= tf.to_float(M)
		#style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		#style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		#total_style_loss += style_loss
	#total_style_loss =tf.to_float(total_style_loss)
	ptint("Not implmented at all")
	total_style_loss = 0.0
	return(total_style_loss)	

def sum_total_variation_losses(sess, net):
	"""
	denoising loss function, this function come from : 
	https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
	"""
	x = net['input']
	weight_help_convergence = 10**9
	[b, h, w, d] = x.get_shape()
	b, h, w, d = tf.to_int32(b),tf.to_int32(h),tf.to_int32(w),tf.to_int32(d)
	tv_y_size = tf.to_float(b * (h-1) * w * d) # Nombre de pixels
	tv_x_size = tf.to_float(b * h * (w-1) * d)
	loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:]) 
	loss_y /= tv_y_size
	loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
	loss_x /= tv_x_size
	loss = 2 * weight_help_convergence * (loss_y + loss_x)
	loss = tf.cast(loss, tf.float32)
	return(loss)
		
def gram_matrix(x,N,M):
  """
  Computation of the Gram Matrix for one layer we normalize with the 
  number of pixels M
  
  Warning the way to compute the Gram Matrix is different from the paper
  but it is equivalent, we use here the F matrix with the shape M*N
  That's quicker
  """
  # The implemented version is quicker than this one :
  #x = tf.transpose(x,(0,3,1,2))
  #F = tf.reshape(x,[tf.to_int32(N),tf.to_int32(M)])
  #G = tf.matmul(F,tf.transpose(F))
  
  F = tf.reshape(x,[M,N])
  G = tf.matmul(tf.transpose(F),F)
  G /= tf.to_float(M)
  # That come from Control paper
  return(G)
 
def get_Gram_matrix(vgg_layers,image_style,pooling_type='avg',padding='SAME'):
	"""
	Computation of all the Gram matrices from one image thanks to the 
	vgg_layers
	"""
	net = net_preloaded(vgg_layers, image_style,pooling_type,padding) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	dict_gram = {}
	for layer in VGG19_LAYERS:
		kind = layer[:4]
		if(kind == 'conv'): 
			a = net[layer]
			_,height,width,N = a.shape
			M = height*width
			A = gram_matrix(a,tf.to_int32(N),tf.to_int32(M)) #  TODO Need to divided by M ????
			dict_gram[layer] = sess.run(A) # Computation
	sess.close()
	tf.reset_default_graph() # To clear all operation and variable
	return(dict_gram)        
		 
def get_features_repr(vgg_layers,image_content,pooling_type='avg',padding='SAME'):
	"""
	Compute the image content representation values according to the vgg
	19 net
	"""
	net = net_preloaded(vgg_layers, image_content,pooling_type,padding) # net for the content image
	sess = tf.Session()
	sess.run(net['input'].assign(image_content))
	dict_features_repr = {}
	for layer in VGG19_LAYERS:
		kind = layer[:4]
		if(kind == 'conv'): 
			P = sess.run(net[layer])
			dict_features_repr[layer] = P # Computation
	sess.close()
	tf.reset_default_graph() # To clear all operation and variable
	return(dict_features_repr)  
	
	
def preprocess(img):
	"""
	This function takes a RGB image and process it to be used with 
	tensorflow
	"""
	# shape (h, w, d) to (1, h, w, d)
	img = img[np.newaxis,:,:,:]
	
	# subtract the imagenet mean for a RGB image
	img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # In order to have channel = (channel - mean) / std with std = 1
	# The input images should be zero-centered by mean pixel (rather than mean image) 
	# subtraction. Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].
	# From https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
	try:
		img = img[...,::-1] # rgb to bgr
	except IndexError:
		raise
	# Both VGG-16 and VGG-19 were trained using Caffe, and Caffe uses OpenCV to 
	# load images which uses BGR by default, so both VGG models are expecting BGR images.
	# It is the case for the .mat save we are using here.
	
	return(img)

def postprocess(img):
	"""
	To the unprocessing analogue to the "preprocess" function from 4D array
	to RGB image
	"""
	# bgr to rgb
	img = img[...,::-1]
	# add the imagenet mean 
	img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
	# shape (1, h, w, d) to (h, w, d)
	img = img[0]
	img = np.clip(img,0,255).astype('uint8')
	return(img)  

def get_M_dict(image_h,image_w):
	"""
	This function compute the size of the different dimension in the con
	volutionnal net
	"""
	M_dict =  {'conv1' : 0,'conv2' : 0,'conv3' : 0,'conv4': 0,'conv5' : 0}
	M = image_h * image_w # Depend on the image size
	image_h_tmp = image_h
	image_w_tmp = image_w
	for key in M_dict.keys():
		M_dict[key] = M
		image_h_tmp =  math.ceil(image_h_tmp / 2)
		image_w_tmp = math.ceil(image_w_tmp / 2)
		M = image_h_tmp*image_w_tmp
	return(M_dict)  
	
def print_loss_tab(sess,list_loss,list_loss_name):
	strToPrint = ''
	for loss,loss_name in zip(list_loss,list_loss_name):
		loss_tmp = sess.run(loss)
		strToPrint +=  loss_name + ' = {:.2e}, '.format(loss_tmp)
	print(strToPrint)
	
def print_loss(sess,loss_total,content_loss,style_loss):
	loss_total_tmp = sess.run(loss_total)
	content_loss_tmp = sess.run(content_loss)
	style_loss_tmp = sess.run(style_loss)
	strToPrint ='Total loss = {:.2e}, Content loss  = {:.2e}, Style loss  = {:.2e}'.format(loss_total_tmp,content_loss_tmp,style_loss_tmp)
	print(strToPrint)

def get_init_noise_img(image_content,init_noise_ratio):
	_,image_h, image_w, number_of_channels = image_content.shape 
	noise_img = np.random.uniform(0,255, (image_h, image_w, number_of_channels)).astype('float32')
	noise_img = preprocess(noise_img)
	noise_img = init_noise_ratio* noise_img + (1.-init_noise_ratio) * image_content
	return(noise_img)

def get_lbfgs_bnds(init_img):
	"""
	This function create the bounds for the LBFGS scipy wrappper, for a 
	image centered according to the ImageNet mean
	"""
	dim1,height,width,N = init_img.shape
	bnd_inf = clip_value_min*np.ones((dim1,height,width,N)).flatten() 
	# We need to flatten the array in order to use it in the LBFGS algo
	bnd_sup = clip_value_max*np.ones((dim1,height,width,N)).flatten()
	bnds = np.stack((bnd_inf, bnd_sup),axis=-1)
	assert len(bnd_sup) == len(init_img.flatten()) # Check if the dimension is right
	assert len(bnd_inf) == len(init_img.flatten()) 
	# Bounds from [0,255] - [124,103]
	# Test
	x0 = np.asarray(init_img).ravel()
	n, = x0.shape
	if len(bnds) != n:
		print("n",n,"len(bnds)",len(bnds))
		print("Erreur a venir")
	return(bnds)

def get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type='avg',padding='SAME'):
	_,image_h_art, image_w_art, _ = image_style.shape
	data_style_path = args.data_folder + "gram_"+args.style_img_name+"_"+str(image_h_art)+"_"+str(image_w_art)+"_"+str(pooling_type)+"_"+str(padding)+".pkl"
	try:
		dict_gram = pickle.load(open(data_style_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The Gram Matrices doesn't exist, we will generate them.")
		dict_gram = get_Gram_matrix(vgg_layers,image_style,pooling_type,padding)
		with open(data_style_path, 'wb') as output_gram_pkl:
			pickle.dump(dict_gram,output_gram_pkl)
		if(args.verbose): print("Pickle dumped")
	return(dict_gram)

def get_features_repr_wrap(args,vgg_layers,image_content,pooling_type='avg',padding='SAME'):
	_,image_h, image_w, number_of_channels = image_content.shape 
	data_content_path = args.data_folder +args.content_img_name+"_"+str(image_h)+"_"+str(image_w)+"_"+str(pooling_type)+"_"+str(padding)+".pkl"
	try:
		dict_features_repr = pickle.load(open(data_content_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The dictionnary of features representation of content image doesn't exist, we will generate it.")
		dict_features_repr = get_features_repr(vgg_layers,image_content,pooling_type,padding)
		with open(data_content_path, 'wb') as output_content_pkl:
			pickle.dump(dict_features_repr,output_content_pkl)
		if(args.verbose): print("Pickle dumped")
		
	return(dict_features_repr)

def plot_image_with_postprocess(args,image,name="",fig=None):
	if(fig==None):
		fig = plt.figure()
	plt.imshow(postprocess(image))
	plt.title(name)
	if(args.verbose): print("Plot",name)
	fig.canvas.flush_events()
	time.sleep(10**(-6))
	return(fig)

def get_init_img_wrap(args,output_image_path,image_content):
	if(not(args.start_from_noise)):
		try:
			init_img = preprocess(scipy.misc.imread(output_image_path).astype('float32'))
		except(FileNotFoundError):
			if(args.verbose): print("Former image not found, use of white noise mixed with the content image as initialization image")
			# White noise that we use at the beginning of the optimization
			init_img = get_init_noise_img(image_content,args.init_noise_ratio)
	else:
		init_img = get_init_noise_img(image_content,args.init_noise_ratio)

	if(args.plot):
		plot_image_with_postprocess(args,init_img.copy(),"Initial Image")
		
	return(init_img)

def load_img(args,img_name):
	"""
	This function load the image and convert it to a numpy array and do 
	the preprocessing
	"""
	image_path = args.img_folder + img_name +args.img_ext
	new_img_ext = args.img_ext
	try:
		img = scipy.misc.imread(image_path)  # Float between 0 and 255
	except IOError:
		if(args.verbose): print("Exception when we try to open the image, try with a different extension format",str(args.img_ext))
		if(args.img_ext==".jpg"):
			new_img_ext = ".png"
		elif(args.img_ext==".png"):
			new_img_ext = ".jpg"
		try:
			image_path = args.img_folder + img_name +new_img_ext # Try the new path
			img = scipy.misc.imread(image_path,mode='RGB')
			if(args.verbose): print("The image have been sucessfully loaded with a different extension")
		except IOError:
			if(args.verbose): print("Exception when we try to open the image, we already test the 2 differents extension.")
			raise
	if(len(img.shape)==2):
		img = gray2rgb(img) # Convertion greyscale to RGB
	img = preprocess(img.astype('float32'))
	return(img)

def get_losses(args,sess, net, dict_features_repr,M_dict,image_style,dict_gram,pooling_type,padding):
	""" Compute the total loss map of the sub loss """
	loss_total = tf.constant(0.)
	list_loss =  []
	list_loss_name =  []
	assert len(args.loss)
	if('Gatys' in args.loss) or ('content'  in args.loss) or ('full' in args.loss):
		content_loss = args.content_strengh * sum_content_losses(sess, net, dict_features_repr,M_dict) # alpha/Beta ratio 
		list_loss +=  [content_loss]
		list_loss_name +=  ['content_loss']
	if('Gatys' in args.loss) or ('texture'  in args.loss) or ('full' in args.loss):
		style_loss = sum_style_losses(sess,net,dict_gram,M_dict)
		list_loss +=  [style_loss]
		list_loss_name +=  ['style_loss']
	if('4moments' in args.loss):
		style_stats_loss = sum_style_stats_loss(sess,net,image_style,M_dict)
		list_loss +=  [style_stats_loss]
		list_loss_name +=  ['style_stats_loss']
	if('InterScale'  in args.loss) or ('full' in args.loss):
		 inter_scale_loss = loss_crosscor_inter_scale(sess,net,image_style,M_dict,sampling=args.sampling,pooling_type=pooling_type)
		 list_loss +=  [inter_scale_loss]
		 list_loss_name +=  ['inter_scale_loss']
	if('nmoments'  in args.loss) or ('full' in args.loss):
		loss_n_moments_val = loss_n_stats(sess,net,image_style,M_dict,args.n,TypeOfComputation='moments')
		list_loss +=  [loss_n_moments_val]
		list_loss_name +=  ['loss_n_moments_val with (n = '+str(args.n)+')']	 
	if('Lp'  in args.loss) or ('full' in args.loss):
		loss_L_p_val =  loss_n_stats(sess,net,image_style,M_dict,args.p,TypeOfComputation='Lp')
		list_loss +=  [loss_L_p_val]
		list_loss_name +=  ['loss_L_p_val with (p = '+str(args.p)+')']	
	if('TV'  in args.loss) or ('full' in args.loss):
		tv_loss =  sum_total_variation_losses(sess, net)
		list_loss +=  [tv_loss]
		list_loss_name +=  ['tv_loss']
	if('autocorr'  in args.loss) or ('full' in args.loss):
		 autocorr_loss = loss_autocorr(sess,net,image_style,M_dict)
		 list_loss +=  [autocorr_loss]
		 list_loss_name +=  ['autocorr_loss']	
	if(args.type_of_loss=='add'):
		loss_total = tf.reduce_sum(list_loss)
	elif(args.type_of_loss=='max'):
		loss_total = tf.reduce_max(list_loss)
	elif(args.type_of_loss=='mul'):
		# If one of the sub loss is zero the total loss is zero !
		if(args.verbose): print("Mul for the total loss : If one of the sub loss is zero the total loss is zero.")
		loss_total = tf.constant(1.)
		for loss in list_loss:
			loss_total *= (loss*10**(-9)) 
	elif(args.type_of_loss=='Keeney'):
		if(args.verbose): print("Keeney for the total loss : they are a lot of different weight everywhere.")
		loss_total = tf.constant(1.*10**9)
		for loss in list_loss:
			loss_total *= (loss*10**(-9) + 1.) 
	 	#Seem to optimize quickly but is stuck
	else:
		if(args.verbose): print("The loss aggregation function is not known")
	list_loss +=  [loss_total]
	list_loss_name +=  ['loss_total']
	return(loss_total,list_loss,list_loss_name)
	
def style_transfer(args,pooling_type='avg',padding='VALID'):
	if args.verbose:
		tinit = time.time()
		print("verbosity turned on")
	
	output_image_path = args.img_folder + args.output_img_name +args.img_ext
	image_content = load_img(args,args.content_img_name)
	image_style = load_img(args,args.image_style_name)
	_,image_h, image_w, number_of_channels = image_content.shape 
	M_dict = get_M_dict(image_h,image_w)
	
	if(args.plot):
		plt.ion()
		plot_image_with_postprocess(args,image_content.copy(),"Content Image")
		plot_image_with_postprocess(args,image_style.copy(),"Style Image")
		fig = None # initialization for later
		
	# TODO add something that reshape the image 
	t1 = time.time()
	vgg_layers = get_vgg_layers()
	
	# Precomputation Phase :
	dict_gram = get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
	dict_features_repr = get_features_repr_wrap(args,vgg_layers,image_content,pooling_type,padding)

	net = net_preloaded(vgg_layers, image_content,pooling_type,padding) # The output image as the same size as the content one
	
	t2 = time.time()
	if(args.verbose): print("net loaded and gram computation after ",t2-t1," s")

	try:
		config = tf.ConfigProto()
		if(args.gpu_frac <= 0.):
			config.gpu_options.allow_growth = True
			if args.verbose: print("Memory Growth")
		elif(args.gpu_frac <= 1.):
			config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
			if args.verbose: print("Becareful args.gpu_frac = ",args.gpu_frac,"It may cause problem if the value is superior to the available memory place.")
		sess = tf.Session(config=config)

		init_img = get_init_img_wrap(args,output_image_path,image_content)
		
		loss_total,list_loss,list_loss_name = get_losses(args,sess, net, dict_features_repr,M_dict,image_style,dict_gram,pooling_type,padding)
			
		# Preparation of the assignation operation
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		placeholder_clip = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)
		clip_op = tf.clip_by_value(placeholder_clip,clip_value_min=clip_value_min,clip_value_max=clip_value_max,name="Clip")
		
		if(args.verbose): print("init loss total")

		if(args.optimizer=='adam'): # Gradient Descent with ADAM algo
			optimizer = tf.train.AdamOptimizer(args.learning_rate)
		elif(args.optimizer=='GD'): # Gradient Descente 
			if((args.learning_rate > 1) and (args.verbose)): print("We recommande you to use a smaller value of learning rate when using the GD algo")
			optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
			
		if((args.optimizer=='GD') or (args.optimizer=='adam')):
			train = optimizer.minimize(loss_total)

			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})
						
			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()") 

			t3 = time.time()
			if(args.verbose): print("sess Adam initialized after ",t3-t2," s")
			# turn on interactive mode
			if(args.verbose): print("loss before optimization")
			if(args.verbose): print_loss_tab(sess,list_loss,list_loss_name)
			for i in range(args.max_iter):
				if(i%args.print_iter==0):
					if(args.tf_profiler):
						run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
						run_metadata = tf.RunMetadata()
						sess.run(train,options=run_options, run_metadata=run_metadata)
						# Create the Timeline object, and write it to a json
						tl = timeline.Timeline(run_metadata.step_stats)
						ctf = tl.generate_chrome_trace_format()
						if(args.verbose): print("Time Line generated")
						nameFile = 'timeline'+str(i)+'.json'
						with open(nameFile, 'w') as f:
							if(args.verbose): print("Save Json tracking")
							f.write(ctf)
							# Read with chrome://tracing
					else:
						t3 =  time.time()
						sess.run(train)
						t4 = time.time()
						result_img = sess.run(net['input'])
						if(args.clip_var==1): # Clipping the variable
							cliptensor = sess.run(clip_op,{placeholder_clip: result_img})
							sess.run(assign_op, {placeholder: cliptensor})
						if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
						if(args.verbose): print_loss_tab(sess,list_loss,list_loss_name)
						if(args.plot): fig = plot_image_with_postprocess(args,result_img,"Intermediate Image",fig)
						result_img_postproc = postprocess(result_img)
						scipy.misc.toimage(result_img_postproc).save(output_image_path)
				else:
					# Just training
					sess.run(train)
					if(args.clip_var==1): # Clipping the variable
						result_img = sess.run(net['input'])
						cliptensor = sess.run(clip_op,{placeholder_clip: result_img})
						sess.run(assign_op, {placeholder: cliptensor}) 
		elif(args.optimizer=='lbfgs'):
			# LBFGS seem to require more memory than Adam optimizer
			
			bnds = get_lbfgs_bnds(init_img)
			# TODO : be able to detect of print_iter > max_iter and deal with it
			nb_iter = args.max_iter  // args.print_iter
			max_iterations_local = args.max_iter // nb_iter
			if(args.verbose): print("Start LBFGS optim with a print each ",max_iterations_local," iterations")
			optimizer_kwargs = {'maxiter': max_iterations_local,'maxcor': args.maxcor}
			optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
				method='L-BFGS-B',options=optimizer_kwargs)         
			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})
						
			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()") 
			
			if(args.verbose): print("loss before optimization")
			if(args.verbose): print_loss_tab(sess,list_loss,list_loss_name)
			for i in range(nb_iter):
				t3 =  time.time()
				optimizer.minimize(sess)
				t4 = time.time()
				if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
				if(args.verbose): print_loss_tab(sess,list_loss,list_loss_name)
				result_img = sess.run(net['input'])
				if(args.plot): fig = plot_image_with_postprocess(args,result_img.copy(),"Intermediate Image",fig)
				result_img_postproc = postprocess(result_img)
				scipy.misc.toimage(result_img_postproc).save(output_image_path)
			
		# The last iterations are not made
		# The End : save the resulting image
		result_img = sess.run(net['input'])
		if(args.plot): plot_image_with_postprocess(args,result_img.copy(),"Final Image",fig)
		result_img_postproc = postprocess(result_img)
		scipy.misc.toimage(result_img_postproc).save(output_image_path)     
		
	except:
		if(args.verbose): print("Error, in the lbfgs case the image can be strange and incorrect")
		result_img = sess.run(net['input'])
		result_img_postproc = postprocess(result_img)
		output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
		scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		# In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
		raise 
	finally:
		sess.close()
		if(args.verbose): 
			print("Close Sess")
			tend = time.time()
			print("Computation total for ",tend-tinit," s")
	if(args.plot): input("Press enter to end and close all")

def main():
	#global args
	parser = get_parser_args()
	args = parser.parse_args()
	style_transfer(args)

def main_with_option():
	parser = get_parser_args()
	#image_style_name= "StarryNight"
	image_style_name = "GrungeMarbled0021_S"
	content_img_name  = "GrungeMarbled0021_S"
	#content_img_name  = "Louvre"
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 1.0 # TODO add a gaussian noise on the image instead a uniform one
	content_strengh = 0.001
	optimizer = 'adam'
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	# In order to set the parameter before run the script
	parser.set_defaults(image_style_name=image_style_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer,maxcor=maxcor,
		learning_rate=learning_rate,sampling=sampling)
	args = parser.parse_args()
	pooling_type='avg'
	padding='SAME'
	style_transfer(args,pooling_type,padding)

if __name__ == '__main__':
	main_with_option()
	#plt.ion()
	#parser = get_parser_args()
	#args = parser.parse_args()
	#image_style_path = args.img_folder + args.image_style_name + args.img_ext
	#image_style = preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	#plot_image_with_postprocess(args,image_style.copy(),"Input Image")
	#sess = tf.Session()
	#augmented = sess.run(get_img_2pixels_more(image_style))
	#print(image_style.shape,augmented.shape)
	#plot_image_with_postprocess(args,augmented.copy(),"augmented Image")
	#sess.close()
	#input("Press Enter to continue...")

	
	

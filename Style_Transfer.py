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
#layers   = [2 5 10 19 28]; for texture generation
# Max et min value from the ImageNet databse mean
clip_value_min=-124
clip_value_max=152

# TODO segment the vgg loader, and the restfautoco
content_layers = [('conv4_2',1)]
style_layers = [('conv1_1',1),('conv2_1',1),('conv3_1',1)]
#style_layers = [('conv1_1',1)]
#style_layers = [('relu1_1',1),('relu2_1',1),('relu3_1',1)]
#style_layers = [('conv1_1',1),('conv2_1',1),('conv3_1',1),('conv4_1',1),('conv5_1',1)]
style_layers = [('conv1_1',1),('pool1',1),('pool2',1),('pool3',1),('pool4',1)]
#style_layers = [('conv3_1',1)]

style_layers_size =  {'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
# TODO : check if the N value are right for the poolx

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
		print("You can download it here : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat")
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
		print(layer,N,M)
		G = gram_matrix(x,N,M) # Nota Bene : the Gram matrix is normalized by M
		style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
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

def pgcd(a,b) :  
	while a%b != 0 : 
		a, b = b, a%b 
	return b

def loss_autocorr(sess,net,image_style,M_dict):
	"""
	Computation of the autocorrelation of the filters
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = 10**9
	total_style_loss = 0.

	sess.run(net['input'].assign(image_style))  
	for layer, weight in style_layers:
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		F_x = tf.fft2d(tf.complex(x,0.))
		R_x = tf.real(tf.multiply(F_x,tf.conj(F_x)))
		R_x /= tf.to_float(M)
		F_a = tf.fft2d(tf.complex(a,0.))
		R_a = tf.real(tf.multiply(F_a,tf.conj(F_a)))
		R_a /= tf.to_float(M)
		style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)

def loss_autocorr2(sess,net,image_style,M_dict):
	"""
	Computation of the autocorrelation of the filters
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = (10**9)
	total_style_loss = 0.
	
	_, h_a, w_a, N = image_style.shape
	#_, h_x, w_x,_ = net['input'].get_shape()
	#if(not(tf.to_int32(h_a) == tf.to_int32(h_x)) or not(tf.to_int32(w_a)==tf.to_int32(w_x))):
		## Bilinear interpolation. 
		#print("Bilinear interpolation. ")
		#image_style_resize = tf.image.resize_images(image_style, [tf.to_int32(h_x), tf.to_int32(w_x)], method=0, align_corners=False) 
		#sess.run(net['input'].assign(image_style_resize))  
		
	sess.run(net['input'].assign(image_style))
		
	for layer, weight in style_layers:
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		x = tf.transpose(x, [0,3,1,2])
		a = tf.transpose(a, [0,3,1,2])
		F_x = tf.fft2d(tf.complex(x,0.))
		R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) # Module de la transformee de Fourrier : produit terme a terme
		#R_x /= tf.to_float(M**2) # Normalisation du module de la TF
		#R_x /= tf.to_float(M) # Normalisation du module de la TF
		F_a = tf.fft2d(tf.complex(a,0.))
		R_a = tf.real(tf.multiply(F_a,tf.conj(F_a))) # Module de la transformee de Fourrier
		#R_a /= tf.to_float(M**2)
		#R_a /= tf.to_float(M) # Which one to get good result ???? TODO ISSUE divde by M M**2 or don't divide ?
		style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		#diff_F = tf.subtract(F_x,F_a) 
		#style_loss = tf.nn.l2_loss(tf.real(tf.multiply(diff_F,tf.conj(diff_F))))
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		total_style_loss += style_loss
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)
	
def compute_ImagePhaseAlea(sess,net,image_style,M_dict): 
	"""
	Add a random phase to the features of the image style at the last
	layer in the network
	"""
	sess.run(net['input'].assign(image_style))
	image_style_Phase = {}
	layer, weight = style_layers[-1]
	a = sess.run(net[layer])
	b, h_a, w_a, N = a.shape
	zeros = np.zeros((h_a,w_a))
	#zeros = np.zeros(1)
	imag = np.random.uniform(low=0.0, high=2.0*np.pi, size=(h_a,w_a))
	#imag = np.random.uniform(low=0.0, high=2.0*np.pi, size=(1))
	angle = tf.complex(zeros,imag)
	exp = tf.exp(angle)
	#print(sess.run(exp))
	#exptile = tf.tile(exp,tf.to_int32([N])) # TODO : verifier la multiplication par une phase aleatoire des features certainement tres mal applique !!! 
	#exptile = tf.tile(exp,tf.to_int32([h_a*w_a*N]))
	#exptile = tf.reshape( exptile,a.shape)
	#exptile_t = tf.transpose(exptile, [0,3,1,2])
	#exptile_t = tf.cast(exptile_t,tf.complex64)
	exptile_t = tf.cast(exp,tf.complex64)
	#print(sess.run(exptile_t))
	at = tf.transpose(a, [0,3,1,2])
	F_a = tf.fft2d(tf.complex(at,0.))
	#F_a_new_phase = tf.multiply(F_a,exptile_t)
	
	
	output_list = []

	for i in range(N):
		output_list.append(tf.multiply(F_a[0,i,:,:],exptile_t))

	F_a_new_phase = tf.stack(output_list)
	#imF = tf.ifft2d(F_a_new_phase)
	#imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
	image_style_Phase[layer] = F_a_new_phase
	
	#for layer, weight in style_layers:
		#a = sess.run(net[layer])
		#b, h_a, w_a, N = a.shape
		##zeros = np.zeros(h_a*w_a)
		#zeros = np.zeros(1)
		##imag = np.random.uniform(low=0.0, high=2.0*np.pi, size=(h_a*w_a))
		#imag = np.random.uniform(low=0.0, high=2.0*np.pi, size=(1))
		#angle = tf.complex(zeros,imag)
		#exp = tf.exp(angle)
		##exptile = tf.tile(exp,tf.to_int32([N]))
		#exptile = tf.tile(exp,tf.to_int32([h_a*w_a*N]))
		#exptile = tf.reshape( exptile,a.shape)
		#exptile_t = tf.transpose(exptile, [0,3,1,2])
		#exptile_t = tf.cast(exptile_t,tf.complex64)
		#at = tf.transpose(a, [0,3,1,2])
		#F_a = tf.fft2d(tf.complex(at,0.))
		#F_a_new_phase = tf.multiply(F_a,exptile_t)
		##imF = tf.ifft2d(F_a_new_phase)
		##imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
		#image_style_Phase[layer] = F_a_new_phase
	return(image_style_Phase)
	
def loss_PhaseAleatoire(sess,net,image_style,image_style_Phase,M_dict):
	"""
	In this loss function we impose the TF transform to the last layer 
	with a random phase imposed and only the spectrum of the 
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = 10**9
	total_style_loss = 0.
	last_style_layers,_ = style_layers[-1]
	for layer, weight in style_layers:
		if(last_style_layers==layer):
			N = style_layers_size[layer[:5]]
			M = M_dict[layer[:5]]
			x = net[layer]
			xt = tf.transpose(x, [0,3,1,2])
			F_x = tf.fft2d(tf.complex(xt,0.))
			F_a = image_style_Phase[layer]
			diff_F = tf.subtract(F_x,F_a)
			diff_F /= M*N
			module  = tf.real(tf.multiply(diff_F,tf.conj(diff_F)))
			loss = tf.reduce_sum(module) 
			loss *=  weight * weight_help_convergence /(length_style_layers)
			total_style_loss += loss
		elif True:
			sess.run(net['input'].assign(image_style))
			N = style_layers_size[layer[:5]]
			M = M_dict[layer[:5]]
			a = sess.run(net[layer])
			x = net[layer]
			x = tf.transpose(x, [0,3,1,2])
			a = tf.transpose(a, [0,3,1,2])
			F_x = tf.fft2d(tf.complex(x,0.))
			R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) # Module de la transformee de Fourrier : produit terme a terme
			R_x /= tf.to_float(M**2) # Normalisation du module de la TF
			F_a = tf.fft2d(tf.complex(a,0.))
			R_a = tf.real(tf.multiply(F_a,tf.conj(F_a))) # Module de la transformee de Fourrier
			R_a /= tf.to_float(M**2)
			style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
			style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
			total_style_loss += style_loss
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)

def loss_PhaseImpose1(sess,net,image_style,M_dict):
	"""
	TODO !!!
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = 10**9
	total_style_loss = 0.
	last_style_layers,_ = style_layers[-1]
	print(last_style_layers)
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# contrainte sur le module uniquement 
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		x = tf.transpose(x, [0,3,1,2])
		a = tf.transpose(a, [0,3,1,2])
		F_x = tf.fft2d(tf.complex(x,0.))
		F_a = tf.fft2d(tf.complex(a,0.))
	
		if(last_style_layers==layer):
			# Contrainte sur la phase 
			angle_a = angle(F_a)
			#angle_a_shiftdim1 = tf.concat([tf.expand_dims(angle_a[:,-1,:,:],0), angle_a[:,:-1,:,:]], axis=1)
			#angle_a_prim = angle_a - angle_a_shiftdim1
			#angle_a_shiftdim2 = tf.concat([tf.expand_dims(angle_a_prim[:,:,-1,:],axis=2), angle_a_prim[:,:,:-1,:]], axis=2)
			#angle_a_prim = angle_a_prim - angle_a_shiftdim2
			#angle_a_shiftdim3 = tf.concat([tf.expand_dims(angle_a_prim[:,:,:,-1],axis=3), angle_a_prim[:,:,:,:-1]], axis=3)
			#angle_a_prim = angle_a_prim - angle_a_shiftdim3
			angle_x = angle(F_x)
			#angle_x_shiftdim1 = tf.concat([tf.expand_dims(angle_x[:,-1,:,:],0), angle_x[:,:-1,:,:]], axis=1)
			#angle_x_prim = angle_x - angle_x_shiftdim1
			#angle_x_shiftdim2 = tf.concat([tf.expand_dims(angle_x_prim[:,:,-1,:],axis=2), angle_x_prim[:,:,:-1,:]], axis=2)
			#angle_x_prim = angle_x_prim - angle_x_shiftdim2
			#angle_x_shiftdim3 = tf.concat([tf.expand_dims(angle_x_prim[:,:,:,-1],axis=3), angle_x_prim[:,:,:,:-1]], axis=3)
			#angle_x_prim = angle_x_prim - angle_x_shiftdim3
			angle_x /= tf.to_float(M)
			angle_a /= tf.to_float(M)
			style_loss = tf.nn.l2_loss(tf.subtract(angle_x,angle_a))    
			style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
			total_style_loss += style_loss
		elif False: 
			R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) # Module de la transformee de Fourrier : produit terme a terme
			R_x /= tf.to_float(M**2) # Normalisation du module de la TF
			F_a = tf.fft2d(tf.complex(a,0.))
			R_a = tf.real(tf.multiply(F_a,tf.conj(F_a))) # Module de la transformee de Fourrier
			R_a /= tf.to_float(M**2)
			style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
			style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
			total_style_loss += style_loss
			
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)
	
def loss_PhaseImpose(sess,net,image_style,M_dict):
	"""
	TODO !!!
	"""
	print("Here ")
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = 10**9
	total_style_loss = 0.
	last_style_layers,_ = style_layers[0]
	print(last_style_layers)
	sess.run(net['input'].assign(image_style))
	alpha = 10**(13)
	for layer, weight in style_layers:
		# contrainte sur le module uniquement 
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		x_t = tf.transpose(x, [0,3,1,2])
		a_t = tf.transpose(a, [0,3,1,2])
		F_x = tf.fft2d(tf.complex(x_t,0.))
		F_a = tf.fft2d(tf.complex(a_t,0.))
	
		if(last_style_layers==layer):
			## Contrainte sur la phase 
			#angle_a = angle(F_a)
			##angle_a_shiftdim1 = tf.concat([tf.expand_dims(angle_a[:,-1,:,:],0), angle_a[:,:-1,:,:]], axis=1)
			##angle_a_prim = angle_a - angle_a_shiftdim1
			##angle_a_shiftdim2 = tf.concat([tf.expand_dims(angle_a_prim[:,:,-1,:],axis=2), angle_a_prim[:,:,:-1,:]], axis=2)
			##angle_a_prim = angle_a_prim - angle_a_shiftdim2
			##angle_a_shiftdim3 = tf.concat([tf.expand_dims(angle_a_prim[:,:,:,-1],axis=3), angle_a_prim[:,:,:,:-1]], axis=3)
			##angle_a_prim = angle_a_prim - angle_a_shiftdim3
			#angle_x = angle(F_x)
			##angle_x_shiftdim1 = tf.concat([tf.expand_dims(angle_x[:,-1,:,:],0), angle_x[:,:-1,:,:]], axis=1)
			##angle_x_prim = angle_x - angle_x_shiftdim1
			##angle_x_shiftdim2 = tf.concat([tf.expand_dims(angle_x_prim[:,:,-1,:],axis=2), angle_x_prim[:,:,:-1,:]], axis=2)
			##angle_x_prim = angle_x_prim - angle_x_shiftdim2
			##angle_x_shiftdim3 = tf.concat([tf.expand_dims(angle_x_prim[:,:,:,-1],axis=3), angle_x_prim[:,:,:,:-1]], axis=3)
			##angle_x_prim = angle_x_prim - angle_x_shiftdim3
			#angle_x /= tf.to_float(M)
			#angle_a /= tf.to_float(M)
			#style_loss = tf.nn.l2_loss(tf.subtract(angle_x,angle_a))   
			#style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
			#total_style_loss += style_loss
			
			# Correlation de phase ? 
			#innerProd = tf.multiply(F_x,tf.conj(F_a))  # sum(ftIm .* conj(ftRef), 3);
			#innerProd /= M**2
			#module_InnerProd = tf.pow(tf.real(tf.multiply(innerProd,tf.conj(innerProd))),0.5)
			#module_innerProd_less_1  = tf.pow(tf.pow(tf.real(tf.multiply(innerProd,tf.conj(innerProd)))-1,2),0.5)
			#style_loss = tf.reduce_sum(tf.multiply(module_InnerProd,module_innerProd_less_1))
			
			angle_x = tf.real(angle(F_x))
			angle_a = tf.real(angle(F_a))
			fft2_angle_x = tf.fft2d(tf.complex(angle_x,0.))
			fft2_angle_a = tf.fft2d(tf.complex(angle_a,0.))
			R_angle_x = tf.real(tf.multiply(fft2_angle_x,tf.conj(fft2_angle_x)))
			R_angle_a = tf.real(tf.multiply(fft2_angle_a,tf.conj(fft2_angle_a)))
			R_angle_a /= tf.to_float(M**2)
			R_angle_x /= tf.to_float(M**2)
			style_loss = tf.nn.l2_loss(tf.subtract(R_angle_x,R_angle_a))  
			
			#dephase = tf.divide(innerProd,module_InnerProd)
			#ftNew =  tf.multiply(dephase,F_x)
			#imF = tf.ifft2d(ftNew)
			#imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
			#loss = tf.nn.l2_loss(tf.subtract(x,imF)) # sum (x**2)/2
			style_loss *= alpha* weight * weight_help_convergence /((2.*(N**2)*length_style_layers))
			total_style_loss += style_loss
		if True:
			R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) # Module de la transformee de Fourrier : produit terme a terme
			R_x /= tf.to_float(M) # Normalisation du module de la TF
			R_a = tf.real(tf.multiply(F_a,tf.conj(F_a))) # Module de la transformee de Fourrier
			R_a /= tf.to_float(M)
			style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a)) 
			style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
			total_style_loss += style_loss
			
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)    
	

def angle(z):
	"""
	Returns the elementwise arctan of z, choosing the quadrant correctly.

	Quadrant I: arctan(y/x)
	Qaudrant II: π + arctan(y/x) (phase of x<0, y=0 is π)
	Quadrant III: -π + arctan(y/x)
	Quadrant IV: arctan(y/x)

	Inputs:
		z: tf.complex64 or tf.complex128 tensor
	Retunrs:
		Angle of z
	"""
	if z.dtype == tf.complex128:
		dtype = tf.float64
	else:
		dtype = tf.float32
	x = tf.real(z)
	y = tf.imag(z)
	xneg = tf.cast(x < 0.0, dtype)
	yneg = tf.cast(y < 0.0, dtype)
	ypos = tf.cast(y >= 0.0, dtype)

	offset = xneg * (ypos - yneg) * np.pi

	return tf.atan(y / x) + offset

	
def loss_intercorr(sess,net,image_style,M_dict):
	"""
	Computation of the correlation of the filter and the interaction 
	long distance of the features : intercorrelation
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = (10**9)
	total_style_loss = 0.
	
	sess.run(net['input'].assign(image_style))  
	for layer, weight in style_layers:
		print(layer)
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		x = tf.transpose(x, [0,3,1,2])
		a = tf.transpose(a, [0,3,1,2])
		F_x = tf.fft2d(tf.complex(x,0.))
		F_x_conj = tf.conj(F_x)
		F_a = tf.fft2d(tf.complex(a,0.))
		F_a_conj = tf.conj(F_a)
		
		#NN = 2
		#alpha = 10
		#R_x = tf.real(tf.ifft2d(tf.multiply(F_x,F_x_conj)))
		#R_a = tf.real(tf.ifft2d(tf.multiply(F_a,F_a_conj)))
		#R_x /= tf.to_float(M**2)
		#R_a /= tf.to_float(M**2)
		#style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		#style_loss *=  alpha * weight * weight_help_convergence  / (2.*(NN**4)*length_style_layers)
		#total_style_loss += style_loss
		#lenRoll = sess.run(tf.random_uniform(minval=0,maxval=N,dtype=tf.int32,shape=[1])) # Between [minval,maxval)
		#print(lenRoll)
		#F_x = tf.concat([F_x[:,lenRoll:,:,:], F_x[:,:lenRoll,:,:]], axis=1)
		#F_a = tf.concat([F_a[:,lenRoll:,:,:], F_a[:,:lenRoll,:,:]], axis=1)
		#R_x = tf.real(tf.ifft2d(tf.multiply(F_x,F_x_conj)))
		#R_a = tf.real(tf.ifft2d(tf.multiply(F_a,F_a_conj)))
		#R_x /= tf.to_float(M**2)
		#R_a /= tf.to_float(M**2)
		#style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
		#style_loss *=  weight * weight_help_convergence  / (2.*(NN**4)*length_style_layers)
		#total_style_loss += style_loss
		
		#lenRoll = sess.run(tf.random_uniform(minval=0,maxval=N,dtype=tf.int32,shape=[1]))
	
		#print(lenRoll)
		NN = N
		for i in range(NN):
			R_x = tf.real(tf.ifft2d(tf.multiply(F_x,F_x_conj)))
			R_a = tf.real(tf.ifft2d(tf.multiply(F_a,F_a_conj)))
			R_x /= tf.to_float(M**2)
			R_a /= tf.to_float(M**2)
			style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
			style_loss *=  weight * weight_help_convergence  / (2.*(NN**4)*length_style_layers)
			total_style_loss += style_loss
			#F_x = tf.stack([F_x[:,-1,:,:], F_x[:,:-1,:,:]], axis=1)
			F_x = tf.concat([tf.expand_dims(F_x[:,-1,:,:],0), F_x[:,:-1,:,:]], axis=1)
			#F_a = tf.stack([F_a[:,-1,:,:], F_a[:,:-1,:,:]], axis=1)
			F_a = tf.concat([tf.expand_dims(F_a[:,-1,:,:],0), F_a[:,:-1,:,:]], axis=1)
			
	return(total_style_loss)
	
def loss_SpectrumOnFeatures(sess,net,image_style,M_dict):
	"""
	In this loss function we impose the spectrum on each features 
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
	weight_help_convergence = 10**9
	total_style_loss = 0.
	sess.run(net['input'].assign(image_style))  
	for layer, weight in style_layers:
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		a = sess.run(net[layer])
		x = net[layer]
		x_transpose = tf.transpose(x, [0,3,1,2])
		a = tf.transpose(a, [0,3,1,2])
		F_x = tf.fft2d(tf.complex(x_transpose,0.))
		F_a = tf.fft2d(tf.complex(a,0.))
		innerProd = tf.multiply(F_x,tf.conj(F_a))  # sum(ftIm .* conj(ftRef), 3);
		module_InnerProd = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5)
		dephase = tf.divide(innerProd,module_InnerProd)
		ftNew =  tf.multiply(dephase,F_x)
		imF = tf.ifft2d(ftNew)
		imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
		loss = tf.nn.l2_loss(tf.subtract(x,imF)) # sum (x**2)/2
		loss *= weight * weight_help_convergence /(M*3*(2.*(N**2)*length_style_layers))
		total_style_loss += loss
	total_style_loss =tf.to_float(total_style_loss)
	return(total_style_loss)
	
	
def loss_fft3D(sess,net,image_style,M_dict):
	"""
	Computation of the 3-dimensional discrete Fourier Transform over the 
	inner-most 3 dimensions of input i.e. height,width,channel :) 
	"""
	# TODO : change the M value attention !!! different size between a and x maybe 
	length_style_layers_int = len(style_layers)
	length_style_layers = float(length_style_layers_int)
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
		F_x = tf.fft3d(tf.complex(x,0.))
		#print(F_x.shape)
		R_x = tf.real(tf.multiply(F_x,tf.conj(F_x)))
		R_x /= tf.to_float(M)
		#print(R_x.shape)
		F_a = tf.fft3d(tf.complex(a,0.))
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
	eps = 0.001
	beta2 = 10**3
	weight_amp = (10**9)*beta2
	M = M_dict['conv1']
	x = net['input']
	a = tf.transpose(image_style, [0,3,1,2])
	x_t = tf.transpose(x, [0,3,1,2])
	F_x = tf.fft2d(tf.complex(x_t,0.))
	#F_a = tf.reduce_sum(tf.fft2d(tf.complex(a,0.)),1, keep_dims=True)
	F_a = tf.fft2d(tf.complex(a,0.))
	#innerProd = tf.reduce_sum( tf.multiply(F_x,tf.conj(F_a)), 1, keep_dims=True )  # sum(ftIm .* conj(ftRef), 3);
	innerProd = tf.multiply(F_x,tf.conj(F_a))  # sum(ftIm .* conj(ftRef), 3);
	module_InnerProd = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5)
	dephase = tf.divide(innerProd,module_InnerProd+eps)
	ftNew =  tf.multiply(dephase,F_x)
	imF = tf.ifft2d(ftNew)
	imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
	loss = tf.nn.l2_loss(tf.subtract(x,imF)) # sum (x**2)/2
	loss *= weight_amp/(M*3)
	return(loss)    

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
	M_dict =  {'conv1' : 0,'relu1' : 0,'pool1':0,'conv2' : 0,'relu2' : 0,'pool2':0,'conv3' : 0,'relu3' : 0,'pool3':0,'conv4': 0,'relu4' : 0,'pool4':0,'conv5' : 0,'relu5' : 0,'pool5':0}
	image_h_tmp = image_h
	image_w_tmp = image_w
	M = image_h_tmp*image_w_tmp
	for key in M_dict.keys():
		if(key[:4]=='conv'):
			M_dict[key] = M
		elif(key[:4]=='pool'):
			image_h_tmp =  math.ceil(image_h_tmp / 2)
			image_w_tmp = math.ceil(image_w_tmp / 2)
			M = image_h_tmp*image_w_tmp
			M_dict[key] = M
		elif(key[:4]=='relu'):
			M_dict[key] = M
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
	""" This function return a white noise image for the initialisation 
	this image can be linearly miwed with the image content with a ratio
	"""
	_,image_h, image_w, number_of_channels = image_content.shape 
	low = 0
	high = 255
	noise_img = np.random.uniform(low,high, (image_h, image_w, number_of_channels)).astype('float32')
	noise_img = preprocess(noise_img)
	if(init_noise_ratio >= 1.):
		noise_img = noise_img
	elif(init_noise_ratio <= 0.0):
		noise_img = image_content
	else:
		noise_img = init_noise_ratio* noise_img + (1.-init_noise_ratio) * image_content
	return(noise_img)
	
def get_init_noise_img_smooth_grad(image_content):
	"""
	This function return a random initial image with a mean near to the 
	mean value of the content image and with a smooth gradient 
	"""
	from skimage import filters
	_,image_h, image_w, number_of_channels = image_content.shape 
	low = -1
	high = 1
	noise_img = np.random.uniform(low,high, (image_h, image_w, number_of_channels))
	gaussian_noise_img = filters.gaussian(noise_img, sigma=2,mode='reflect')
	for i in range(3):
		 gaussian_noise_img[:,:,i] += np.mean(image_content[:,:,i]) # Add the mean of each channel
	gaussian_noise_img = np.clip(gaussian_noise_img,0.,255.)
	preprocess_img = preprocess(gaussian_noise_img)
	return(preprocess_img)

def get_lbfgs_bnds(init_img):
	"""
	This function create the bounds for the LBFGS scipy wrappper, for a 
	image centered according to the ImageNet mean
	"""
	# TODO : better bounds with respect the ImageNet mean 
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
		if(args.verbose): print("Load Data ",data_style_path)
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
	"""
	Plot the image using matplotlib
	"""
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
	elif(args.smooth_grad):
		if(args.verbose): print("Noisy image generation with a smooth gradient")
		init_img = get_init_noise_img_smooth_grad(image_content)
	else:
		if(args.verbose): print("Noisy image generation init_noise_ratio = ",args.init_noise_ratio)
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
	if('fft3D'  in args.loss) or ('full' in args.loss):
		 fft3D_loss = loss_fft3D(sess,net,image_style,M_dict)
		 list_loss +=  [fft3D_loss]
		 list_loss_name +=  ['fft3D_loss']  
	if('spectrum'  in args.loss) or ('full' in args.loss):
		 spectrum_loss = loss_spectrum(sess,net,image_style,M_dict)
		 list_loss +=  [spectrum_loss]
		 list_loss_name +=  ['spectrum_loss']    
	if('SpectrumOnFeatures'  in args.loss) or ('full' in args.loss):
		 SpectrumOnFeatures_loss = loss_SpectrumOnFeatures(sess,net,image_style,M_dict)
		 list_loss +=  [SpectrumOnFeatures_loss]
		 list_loss_name +=  ['SpectrumOnFeatures_loss'] 
	if('phaseAlea' in args.loss) or ('full' in args.loss):
		 image_style_Phase = compute_ImagePhaseAlea(sess,net,image_style,M_dict)
		 phaseAlea_loss = loss_PhaseAleatoire(sess,net,image_style,image_style_Phase,M_dict)
		 list_loss +=  [phaseAlea_loss]
		 list_loss_name +=  ['phaseAlea_loss']  
	if('intercorr' in args.loss) or ('full' in args.loss):
		 print("With do a Ressource Exhausted error")
		 intercorr_loss = loss_intercorr(sess,net,image_style,M_dict)
		 list_loss +=  [intercorr_loss]
		 list_loss_name +=  ['intercorr_loss']  
	if('current' in args.loss) or ('full' in args.loss):
		 PhaseImpose_loss = loss_PhaseImpose(sess,net,image_style,M_dict)   
		 list_loss +=  [PhaseImpose_loss]
		 list_loss_name +=  ['PhaseImpose_loss']          
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
		print("style",args.style_img_name,"content",args.content_img_name)
	
	output_image_path = args.img_folder + args.output_img_name + '.jpg' #args.img_ext
	output_image_pathpng = args.img_folder + args.output_img_name + '.png' #args.img_ext
	image_content = load_img(args,args.content_img_name)
	image_style = load_img(args,args.style_img_name)
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
	print(args)
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
			## FYI that we've now added a var_to_bounds argument to ScipyOptimizerInterface that allows specifying per-Variable bounds. 
			# It's submitted in the internal Google repository so should be available in GitHub/PyPi soon. 
			# Also be aware that once the update is rolled out, supplying the bounds keyword explicitly as I suggested above will raise an exception...
			# TODO change that when you will update tensorflow
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
				#scipy.misc.toimage(result_img_postproc).save(output_image_path)
				scipy.misc.imsave(output_image_path,result_img_postproc)
				#scipy.misc.toimage(result_img_postproc).save(output_image_pathpng)
				scipy.misc.imsave(output_image_pathpng,result_img_postproc)
				#scipy.misc.toimage(result_img_postproc,mode='P').save(output_image_path, format='JPEG', subsampling=0, quality=100)
				#import cv2
				#cv2.imwrite("P1.png", result_img_postproc)
				

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
	image_style_name= "StarryNight_Big"
	image_style_name= "StarryNight"
	starry = "StarryNight"
	marbre = 'GrungeMarbled0021_S'
	tile =  "TilesOrnate0158_1_S"
	tile2 = "TilesZellige0099_1_S"
	peddle = "pebbles"
	brick = "BrickSmallBrown0293_1_S"
	image_style_name= brick
	content_img_name  = brick
	#content_img_name  = "Louvre"
	max_iter = 2000
	print_iter = 200
	start_from_noise = 1 # True
	init_noise_ratio = 1.0 # TODO add a gaussian noise on the image instead a uniform one
	content_strengh = 0.001
	optimizer = 'adam'
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	# In order to set the parameter before run the script
	parser.set_defaults(style_img_name=image_style_name,max_iter=max_iter,
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
	# Use CUDA_VISIBLE_DEVICES='' python ... to avoid using CUDA
	# Pour update Tensorflow : python3.6 -m pip install --upgrade tensorflow-gpu
	
	

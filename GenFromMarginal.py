#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 1st June 

This script have the goal to generate texture from the marginal of the 
features maps

Bien que le problème soit mal posé nous essayons d'inverser le réseau et
de remonter à l'image d'origine sans faire intervenir d'optimisation

@author: nicolas

"""


import scipy
import numpy as np
import tensorflow as tf
import Style_Transfer as st
from Arg_Parser import get_parser_args 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from tensorflow.python.framework import dtypes
import matplotlib.gridspec as gridspec
import math
from skimage import exposure
from PIL import Image
import Misc
import skimage
import scipy.stats as stats
from activations import lrelu,ilrelu

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

VGG19_LAYERS_INDICES = {'conv1_1' : 0,'conv1_2' : 2,'conv2_1' : 5,'conv2_2' : 7,
	'conv3_1' : 10,'conv3_2' : 12,'conv3_3' : 14,'conv3_4' : 16,'conv4_1' : 19,
	'conv4_2' : 21,'conv4_3' : 23,'conv4_4' : 25,'conv5_1' : 28,'conv5_2' : 30,
	'conv5_3' : 32,'conv5_4' : 34}

VGG19_LAYERS_INTEREST = (
	'conv1_1','conv2_1', 'conv3_1'
)

leak= 0.01

def net_preloaded(vgg_layers, input_image,relu_kind=None,pooling_type='avg',padding='SAME'):
	"""
	This function read the vgg layers and create the net architecture
	We need the input image to know the dimension of the input layer of the net
	"""
	net = {}
	shapes = {}
	_,height, width, numberChannels = input_image.shape # In order to have the right shape of the input
	current = tf.Variable(np.zeros((1, height, width, numberChannels), dtype=np.float32))
	shapes['input'] = current.get_shape().as_list()
	net['input'] = current
	print('input',current.get_shape())
	for i, name in enumerate(VGG19_LAYERS):
		kind = name[:4]
		shapes[name] = current.get_shape().as_list()
		if(kind == 'conv'):
			#if(VGG19_mat=='texturesyn_normalizedvgg.mat'):
			# Only way to get the weight of the kernel of convolution
			# Inspired by http://programtalk.com/vs2/python/2964/facenet/tmp/vggverydeep19.py/
			kernels = vgg_layers[i][0][0][2][0][0] 
			bias = vgg_layers[i][0][0][2][0][1]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
			bias = tf.constant(bias.reshape(-1))
			current = st.conv_layer(current, kernels, bias,name,padding) 
			# Update the  variable named current to have the right size
		elif(kind == 'relu'):
			if(relu_kind=='l'): # leakly relu
				current = lrelu(current, leak=leak)
			else:
				current = tf.nn.relu(current,name=name)
		elif(kind == 'pool'):
			current = st.pool_layer(current,name,pooling_type,padding)

		net[name] = current
		print(name,current.get_shape())
	assert len(net) == len(VGG19_LAYERS) +1 # Test if the length is right 
	return(net,shapes)

def net_reversed_loaded(vgg_layers,last_layer_image,style_layers,shapes,relu_kind,padding='SAME'):
	"""
	Reversed network to generate texture
	"""
	VGG19_LAYERS_reversed =  VGG19_LAYERS[::-1]
	last_layer = style_layers[-1]
	#print(last_layer)
	net = {}
	net_started = False
	for i,name in enumerate(VGG19_LAYERS_reversed):
		indice = 35-i
		if(name == last_layer):
			_,height_image, width_image, numberChannels = last_layer_image.shape # In order to have the right shape of the input
			current = tf.Variable(np.zeros((1, height_image, width_image, numberChannels), dtype=np.float32))
			net_started = True
			net['input_reversed'] = current
			print('input_reversed',current.get_shape())
		if(net_started):
			shape = shapes[name]
			kind = name[:4]
			if(kind == 'conv'):
				kernels = vgg_layers[indice][0][0][2][0][0] 
				bias = vgg_layers[indice][0][0][2][0][1]
				# matconvnet: weights are [width, height, in_channels, out_channels]
				# tensorflow: weights are [height, width, in_channels, out_channels]
				kernels = np.transpose(kernels, (1,0 ,2, 3))
				height, width, in_channels, out_channels = kernels.shape
				print('shape kernel',height, width, in_channels, out_channels)
				height, width, in_channels, out_channels = kernels_bis.shape
				print('shape kernels_bis',height, width, in_channels, out_channels)
				print('shape',shape)
				kernels = tf.constant(kernels)
				bias = tf.constant(bias.reshape(-1))
				bias = tf.constant(np.zeros(bias.shape).astype('float32'))
				#kernels = tf.constant_initializer(kernels)
				#bias = tf.constant_initializer(bias.reshape(-1))
				current_bias =tf.subtract(current , bias)
				current = tf.nn.conv2d_transpose(value=current_bias,filter=kernels,output_shape=tf.stack([1, shape[1], shape[2], shape[3]]),
				strides=[1, 1, 1, 1], padding=padding) # En fait la convolution transpose n'est qu'une reconvolution et non une déconvolution !!! 
			elif(kind == 'relu'):
				if(relu_kind=='l'): # leakly relu
					current = ilrelu(current, leak=leak)
				else:
					current = tf.nn.relu(current,name=name)
					#current = current
			elif(kind == 'pool'):
				#current = un_pool_layer(current)
				#print("before pooling",current.get_shape(),2*height_image,2*width_image)
				current = tf.image.resize_images(current,[2*height_image,2*width_image])
				#print("after pooling",current.get_shape())
				height_image  = 2*height_image
				width_image = 2*width_image
			net[name] = current
			print(name,current.get_shape())
	return(net)


def net_inversed_loaded(vgg_layers,last_layer_image,style_layers,shapes,relu_kind,padding='SAME'):
	"""
	Reversed network to generate texture
	"""
	VGG19_LAYERS_reversed =  VGG19_LAYERS[::-1]
	last_layer = style_layers[-1]
	#print(last_layer)
	net = {}
	net_started = False
	for i,name in enumerate(VGG19_LAYERS_reversed):
		indice = 35-i
		if(name == last_layer):
			_,height_image, width_image, numberChannels = last_layer_image.shape # In order to have the right shape of the input
			current = tf.Variable(np.zeros((1, height_image, width_image, numberChannels), dtype=np.float32))
			net_started = True
			net['input_reversed'] = current
			print('input_reversed',current.get_shape())
		if(net_started):
			shape = shapes[name]
			kind = name[:4]
			if(kind == 'conv'):
				kernels = vgg_layers[indice][0][0][2][0][0] 
				bias = vgg_layers[indice][0][0][2][0][1]
				# matconvnet: weights are [width, height, in_channels, out_channels]
				# tensorflow: weights are [height, width, in_channels, out_channels]
				kernels = np.transpose(kernels, (1,0 ,2, 3))
				height, width, in_channels, out_channels = kernels.shape
				#_,h,w,_ = current.get_shape()
				#current_bis = tf.zeros([1,h,w,in_channels])
				#kernels_fftn = np.zeros((height, width,out_channels, in_channels))
				#for k in range(out_channels):
					#kernels_fftn = np.fft.fftn(kernels[:,:,:,k],axes=(0,1))
					#kernels_fftn_abs2 = np.power(np.absolute(kernels_fftn),2)
					#kernels_fftn_normed = np.divide(np.conj(kernels_fftn),kernels_fftn_abs2)
					#kernels_inverted = np.real(np.fft.ifftn(kernels_fftn_normed,axes=(0,1))).astype('float32')
					#kernels_inverted = tf.expand_dims(tf.constant(kernels_inverted),axis=2)
					#current_k = tf.expand_dims(current[:,:,:,k],axis=3)
					#conv = tf.nn.conv2d(current_k, kernels_inverted, strides=(1, 1, 1, 1),padding=padding)
					#conv = tf.divide(conv,tf.to_float(out_channels))
					#current_bis = tf.add(current_bis,conv)
				#current = current_bis
				
				print('kernels',kernels.shape)
				kernels_fftn = np.fft.fftn(kernels,axes=(0,1,2))
				print('kernels_fftn',kernels_fftn.shape)
				kernels_fftn_abs2 = np.power(np.absolute(kernels_fftn),2)
				print('kernels_fftn_abs2',kernels_fftn_abs2.shape)
				kernels_fftn_normed = np.divide(np.conj(kernels_fftn),kernels_fftn_abs2)
				kernels_inverted = np.real(np.fft.ifftn(kernels_fftn_normed,axes=(0,1,2))).astype('float32')
				print('kernels_inverted',kernels_inverted.shape)
				print('shape',shape)
				kernels_inverted = np.transpose(kernels_inverted, (0,1 ,3, 2))
				height, width, in_channels, out_channels = kernels_inverted.shape
				print('shape kernels_inverted',height, width, in_channels, out_channels)
				kernels_inverted = tf.constant(kernels_inverted)
				

				bias = tf.constant(bias.reshape(-1))
				bias = tf.constant(np.zeros(bias.shape).astype('float32'))
				current_bias =tf.subtract(current , bias)
				#current = tf.nn.conv2d_transpose(value=current_bias,filter=kernels_inverted,output_shape=tf.stack([1, shape[1], shape[2], shape[3]]),strides=[1, 1, 1, 1], padding=padding)
				print(current_bias.get_shape(),kernels_inverted.get_shape())
				current = tf.nn.conv2d(current_bias, kernels_inverted, strides=(1, 1, 1, 1),padding=padding)
			elif(kind == 'relu'):
				if(relu_kind=='l'): # leakly relu
					current = ilrelu(current, leak=leak)
				else:
					current = tf.nn.relu(current,name=name)
					#current = current
			elif(kind == 'pool'):
				#current = un_pool_layer(current)
				#print("before pooling",current.get_shape(),2*height_image,2*width_image)
				current = tf.image.resize_images(current,[2*height_image,2*width_image])
				#print("after pooling",current.get_shape())
				height_image  = 2*height_image
				width_image = 2*width_image
			name += '_out'
			net[name] = current
			print(name,current.get_shape())
	return(net)	
def un_pool_layer(current):
	_,height, width, numberChannels = current.shape
	new_height = 2*tf.to_int32(height)
	new_width = 2*tf.to_int32(width)
	#new_image = tf.image.resize_images(current,[new_height,new_width])
	new_image = tf.image.resize_bilinear(current,[new_height,new_width])
	print(new_image.get_shape())
	return(new_image)


def generationFromMarginal(args):
	"""
	The goal of this function is to be able to generate a texture directly 
	from the marginals of the filters responses of the networks 
	by inverting the networks directly without optimization 
	"""
	# Info on the nets 
	relu_kind = None
	padding = 'SAME'
	pooling_type='avg'
	
	# Getting the image
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	org_image = scipy.misc.imread(image_style_path).astype('float32')
	image_style = st.preprocess(org_image.copy()) 
	_,image_h_art, image_w_art, channels = image_style.shape 
	
	pltname = 'pdfTest.pdf'
	pp = PdfPages(pltname)
	reshaped_original =np.reshape(org_image,(image_h_art*image_w_art,channels))
	f, ax = plt.subplots(2,3)
	binwidth = 255./50
	for i in range(channels):
		ax[0,i].hist(reshaped_original[:,i],bins=np.arange(0, 255 + binwidth, binwidth))

	# Getting the VGG network
	vgg_layers = st.get_vgg_layers()
	net,shapes = net_preloaded(vgg_layers, image_style,relu_kind,pooling_type,padding) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_style.shape)
	assign_op = net['input'].assign(placeholder)
	args.style_layers = ['conv1_1']
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(assign_op, {placeholder: image_style})
	#sess.graph.finalize()
	last_layer_image = None
	dict_output = {}
	for layer in args.style_layers:
		a = net[layer].eval(session=sess)
		last_layer_image = a
		dict_output[layer] = a
		#Matrix = a[0]
		#h,w,channels = Matrix.shape
		#Matrix_reshaped = np.reshape(Matrix,(h*w,channels))
		#random_features_maps = np.zeros(a.shape)
		#for i in range(channels):
			#samples = Matrix_reshaped[:,i]
			#beta, loc, scale = stats.gennorm.fit(samples)
			#r = stats.gennorm.rvs(beta,loc,scale, size=h*w)
			#r_reshaped = np.reshape(r,(h,w))
			#random_features_maps[0,:,:,i] = r_reshaped
		#last_layer_image = random_features_maps
	
	print(last_layer_image.shape)
	last_layer = args.style_layers[-1]
	net_inversed = net_inversed_loaded(vgg_layers,last_layer_image, args.style_layers,shapes,relu_kind=relu_kind,padding=padding)
	placeholder2 = tf.placeholder(tf.float32, shape=last_layer_image.shape)
	assign_op2 = net_inversed['input_reversed'].assign(placeholder2)
	sess.run(assign_op2, {placeholder2: last_layer_image})
	#for layer in args.style_layers[::-1]:
		##if no
		#a = net_inversed_loaded[layer].eval(session=sess) # net_reversed[layer] retourne ce qu'il y avant la couche layer

	a = net_inversed['conv1_1_out'].eval(session=sess)
	print('conv1_1_out',a.shape,np.max(a),np.min(a),np.std(a))
	#a = a*255./(np.max(a)-np.min(a))
	result_img_postproc = st.postprocess(a) # subtract
	output_image_path = args.img_output_folder + args.output_img_name +args.img_ext
	scipy.misc.toimage(result_img_postproc).save(output_image_path) 

	result_img_postproc_hist = Misc.histogram_matching(result_img_postproc.copy(), org_image.copy(),grey=False, n_bins=200) #  image whose distribution should be remapped and then image that distribution should be matched
	
	output_image_path = args.img_output_folder + args.output_img_name +'_hist' +args.img_ext
	scipy.misc.toimage(result_img_postproc_hist).save(output_image_path) 
	
	reshaped =np.reshape(result_img_postproc,(image_h_art*image_w_art,channels))
	for i in range(channels):
		ax[1,i].hist(reshaped[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
	plt.suptitle('Comparaison Histogram BGR origine Generation')
	plt.savefig(pp, format='pdf')
	plt.close()
	
	f, ax = plt.subplots(2,3)
	reshaped_hist =np.reshape(result_img_postproc_hist,(image_h_art*image_w_art,channels))
	for i in range(channels):
		ax[0,i].hist(reshaped_original[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
		ax[1,i].hist(reshaped_hist[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
	plt.suptitle('Comparaison Histogram BGR origine Generation modif par Hist Matching')
	plt.savefig(pp, format='pdf')

	plt.close()
	pp.close()

def generationFromOriginal(args):
	"""
	The goal of this function is to be able to generate a texture directly 
	from the marginals of the filters responses of the networks 
	by inverting the networks directly without optimization 
	"""
	# Info on the nets 
	relu_kind = None
	padding = 'SAME'
	pooling_type='avg'
	
	# Getting the image
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	org_image = scipy.misc.imread(image_style_path).astype('float32')
	image_style = st.preprocess(org_image.copy()) 
	_,image_h_art, image_w_art, channels = image_style.shape 
	
	pltname = 'pdfTest.pdf'
	pp = PdfPages(pltname)
	reshaped_original =np.reshape(org_image,(image_h_art*image_w_art,channels))
	f, ax = plt.subplots(2,3)
	binwidth = 255./50
	for i in range(channels):
		ax[0,i].hist(reshaped_original[:,i],bins=np.arange(0, 255 + binwidth, binwidth))

	
	
	# Getting the VGG network
	vgg_layers = st.get_vgg_layers()
	net,shapes = net_preloaded(vgg_layers, image_style,relu_kind,pooling_type,padding) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_style.shape)
	assign_op = net['input'].assign(placeholder)
	args.style_layers = ['conv1_1']
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(assign_op, {placeholder: image_style})
	#sess.graph.finalize()
	last_layer_image = None
	for layer in args.style_layers:
		a = net[layer].eval(session=sess)
		last_layer_image = a
		#Matrix = a[0]
		#h,w,channels = Matrix.shape
		#Matrix_reshaped = np.reshape(Matrix,(h*w,channels))
		#random_features_maps = np.zeros(a.shape)
		#for i in range(channels):
			#samples = Matrix_reshaped[:,i]
			#beta, loc, scale = stats.gennorm.fit(samples)
			#r = stats.gennorm.rvs(beta,loc,scale, size=h*w)
			#r_reshaped = np.reshape(r,(h,w))
			#random_features_maps[0,:,:,i] = r_reshaped
		#last_layer_image = random_features_maps
	
	print(last_layer_image.shape)
	last_layer = args.style_layers[-1]
	net_reversed = net_reversed_loaded(vgg_layers,last_layer_image, args.style_layers,shapes,relu_kind=relu_kind,padding=padding)
	placeholder2 = tf.placeholder(tf.float32, shape=last_layer_image.shape)
	assign_op2 = net_reversed['input_reversed'].assign(placeholder2)
	sess.run(assign_op2, {placeholder2: last_layer_image})
	a = net_reversed['conv1_1'].eval(session=sess)
	print(a.shape)
	result_img_postproc = st.postprocess(a) # subtract
	output_image_path = args.img_output_folder + args.output_img_name +args.img_ext
	scipy.misc.toimage(result_img_postproc).save(output_image_path) 

	result_img_postproc_hist = Misc.histogram_matching(result_img_postproc.copy(), org_image.copy(),grey=False, n_bins=200) #  image whose distribution should be remapped and then image that distribution should be matched
	
	output_image_path = args.img_output_folder + args.output_img_name +'_hist' +args.img_ext
	scipy.misc.toimage(result_img_postproc_hist).save(output_image_path) 
	
	reshaped =np.reshape(result_img_postproc,(image_h_art*image_w_art,channels))
	for i in range(channels):
		ax[1,i].hist(reshaped[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
	plt.suptitle('Comparaison Histogram BGR origine Generation')
	plt.savefig(pp, format='pdf')
	plt.close()
	
	f, ax = plt.subplots(2,3)
	reshaped_hist =np.reshape(result_img_postproc_hist,(image_h_art*image_w_art,channels))
	for i in range(channels):
		ax[0,i].hist(reshaped_original[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
		ax[1,i].hist(reshaped_hist[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
	plt.suptitle('Comparaison Histogram BGR origine Generation modif par Hist Matching')
	plt.savefig(pp, format='pdf')
	
	#result_img_postproc_hist2= Misc.histeq(result_img_postproc_hist)
	#f, ax = plt.subplots(2,3)
	#reshaped_hist2 =np.reshape(result_img_postproc_hist2,(image_h_art*image_w_art,channels))
	#for i in range(channels):
		#ax[0,i].hist(reshaped_original[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
		#ax[1,i].hist(reshaped_hist2[:,i],bins=np.arange(0, 255 + binwidth, binwidth))
	#plt.suptitle('Comparaison Histogram BGR origine Generation modif par Hist Matching')
	#plt.savefig(pp, format='pdf')
	
	#output_image_path = args.img_output_folder + args.output_img_name +'_hist2' +args.img_ext
	#scipy.misc.toimage(result_img_postproc_hist2).save(output_image_path) 
	
	plt.close()
	pp.close()
	

if __name__ == '__main__':
	parser = get_parser_args()
	style_img_name = "pebbles"
	style_img_name = "TilesOrnate0158_1_S"
	output_img_name = "Gen"
	parser.set_defaults(verbose=True,style_img_name=style_img_name,output_img_name=output_img_name)
	args = parser.parse_args()
	#generationFromOriginal(args)
	generationFromMarginal(args)
	

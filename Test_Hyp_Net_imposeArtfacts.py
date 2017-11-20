#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 2017

The goal of this script is to test an hypothesis :
Does the algo create the artefacts in the images ?

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
import Misc
import Style_Transfer as st
from numpy import linalg as LA

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3')
# Warning some layers have been removed from the VGG19 layers 	
#layers   = [2 5 10 19 28]; for texture generation
style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
# TODO : check if the N value are right for the poolx

#def style_layer_loss(a, x):
	#z, h, w, d = a.get_shape()
	#M = h.value * w.value
	#N = d.value
	#A = gram_matrix(a, M, N)
	#G = gram_matrix(x, M, N)
	#loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
	#return loss

def gram_matrix(x, area, depth):
	F = tf.reshape(x, (area, depth))
	G = tf.matmul(tf.transpose(F), F)
	return G

def net_preloaded_VALIDTRUE(vgg_layers, input_image,pooling_type='avg'):
	"""
	This function read the vgg layers and create the net architecture
	We need the input image to know the dimension of the input layer of the net
	In this case we use a true valid padding in the function
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
			kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
			bias = tf.constant(bias.reshape(-1))
			current = conv_layer(current, kernels, bias,name)
			# Update the  variable named current to have the right size
		elif(kind == 'relu'):
			current = tf.nn.relu(current,name=name)
		elif(kind == 'pool'):
			current = pool_layer(current,name,pooling_type)

		net[name] = current

	assert len(net) == len(VGG19_LAYERS) +1 # Test if the length is right
	return(net)

def conv_layer(input, weights, bias,name):
	"""
	This function create a conv2d with the already known weight and bias

	conv2d :
	Computes a 2-D convolution given 4-D input and filter tensors.
	input: A Tensor. Must be one of the following types: half, float32, float64
	Given an input tensor of shape [batch, in_height, in_width, in_channels] and
	a filter / kernel tensor of shape
	[filter_height, filter_width, in_channels, out_channels]
	
	Padding = VALID without adding element around
	"""
	stride = 1
	# in order to have a true valid !!!
	conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
		padding='VALID',name=name)
	# We need to impose the weights as constant in order to avoid their modification
	# when we will perform the optimization
	return(tf.nn.bias_add(conv, bias))

def pool_layer(input,name,pooling_type='avg',padding='SAME'):
	"""
	Average pooling on windows 2*2 with stride of 2
	input is a 4D Tensor of shape [batch, height, width, channels]
	Each pooling op uses rectangular windows of size ksize separated by offset
	strides in the avg_pool function
	"""
	stride_pool = 2
	if pooling_type == 'avg':
		pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
				padding=padding,name=name)
	elif pooling_type == 'max':
		pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
				padding=padding,name=name)
	return(pool)

def get_Gram_matrix_wrap_VALIDTRUE(args,vgg_layers,image_style,pooling_type='avg',padding='SAME'):
	_,image_h_art, image_w_art, _ = image_style.shape
	vgg_name = args.vgg_name
	stringAdd = ''
	if(vgg_name=='normalizedvgg.mat'):
		stringAdd = '_n'
	elif(vgg_name=='imagenet-vgg-verydeep-19.mat'):
		stringAdd = '_v' # Regular one
	elif(vgg_name=='random_net.mat'):
		stringAdd = '_r' # random
	data_style_path = args.data_folder + "gramVALIDTRUE_"+args.style_img_name+"_"+str(image_h_art)+"_"+str(image_w_art)+"_"+str(pooling_type)+"_"+str(padding)+stringAdd+".pkl"
	if(vgg_name=='random_net.mat'):
		try:
			os.remove(data_style_path)
		except:
			pass
	try:
		if(args.verbose): print("Load Data ",data_style_path)
		dict_gram = pickle.load(open(data_style_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The Gram Matrices doesn't exist, we will generate them.")
		dict_gram = get_Gram_matrix_VALIDTRUE(vgg_layers,image_style,pooling_type)
		with open(data_style_path, 'wb') as output_gram_pkl:
			pickle.dump(dict_gram,output_gram_pkl)
		if(args.verbose): print("Pickle dumped")
	return(dict_gram)

def style_losses_VALIDTRUE(sess, net, style_image,style_layers):
	"""
	Compute the style term of the loss function with Gram Matrix from the
	Gatys Paper
	Input :
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of Gram Matrices
	- the dictionnary of the size of the image content through the net
	Return an array of the style losses for the diferents layers and the total_style_loss
	"""
	# Info for the vgg19
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 10**(9) # This wight come from a paper of Gatys
	# Because the function is pretty flat
	total_style_loss = 0
	style_losses_tab = []
	sess.run(net['input'].assign(style_image))
	for i, couple in enumerate(style_layers):
		layer, weight = couple
		print(layer)
		#a = sess.run(net[layer])
		a = net[layer].eval(session=sess)
		x = net[layer]
		a = tf.convert_to_tensor(a)
		_, h, w, d = a.get_shape()
		M = h.value * w.value
		N = d.value
		A = gram_matrix(a,N,M)
		G = gram_matrix(x,N,M) # Nota Bene : the Gram matrix is normalized by M
		print("Gram done")
		style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
		style_losses_tab += [style_loss]
		total_style_loss += style_loss
	return(total_style_loss,style_losses_tab)
	return(total_style_loss,style_losses_tab)

def compute_loss_tab(sess,list_loss):
	loss_tab_eval = np.zeros_like(list_loss)
	for i,loss in enumerate(list_loss):
		loss_tab_eval[i] = sess.run(loss)
	return(loss_tab_eval)

def get_Gram_matrix_VALIDTRUE(vgg_layers,image_style,pooling_type='avg'):
	"""
	Computation of all the Gram matrices from one image thanks to the
	vgg_layers
	"""
	dict_gram = {}
	net = net_preloaded_VALIDTRUE(vgg_layers, image_style,pooling_type) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	a = net['input']
	_,height,width,N = a.shape
	M = height*width
	A = gram_matrix(a,tf.to_int32(N),tf.to_int32(M)) #  TODO Need to divided by M ????
	dict_gram['input'] = sess.run(A)
	for layer in VGG19_LAYERS:
		a = net[layer]
		_,height,width,N = a.shape
		M = height*width
		A = gram_matrix(a,tf.to_int32(N),tf.to_int32(M)) #  TODO Need to divided by M ????
		dict_gram[layer] = sess.run(A) # Computation
	sess.close()
	tf.reset_default_graph() # To clear all operation and variable
	return(dict_gram)

def print_bnds(result_img,image_style,clip_value_max,clip_value_min):
	# Code to test the bounds problem :
	maxB = np.max(result_img[:,:,:,0])
	maxG = np.max(result_img[:,:,:,1])
	maxR = np.max(result_img[:,:,:,2])
	minB = np.min(result_img[:,:,:,0])
	minG = np.min(result_img[:,:,:,1])
	minR = np.min(result_img[:,:,:,2])
	print("Synthesis Image",maxB,maxG,maxR,minB,minG,minR)
	maxB = np.max(image_style[:,:,:,0])
	maxG = np.max(image_style[:,:,:,1])
	maxR = np.max(image_style[:,:,:,2])
	minB = np.min(image_style[:,:,:,0])
	minG = np.min(image_style[:,:,:,1])
	minR = np.min(image_style[:,:,:,2])
	print("RefImage",maxB,maxG,maxR,minB,minG,minR)
	print("Clip",clip_value_max,clip_value_min)

def texture_syn_with_loss_decomposition(args,test_bnds = False):
	"""
	The goal is to vizualise with part of the loss is the more important
	"""

	if args.verbose:
		print("verbosity turned on")
		print(args)


	if(args.verbose and args.img_ext=='.jpg'): print("Be careful you are saving the image in JPEG !")
	image_content = st.load_img(args,args.content_img_name)
	image_style = st.load_img(args,args.style_img_name)
	_,image_h, image_w, number_of_channels = image_content.shape
	#M_dict = st.get_M_dict(image_h,image_w)

	if(args.clipping_type=='ImageNet'):
		BGR=False
		clip_value_min,creturnlip_value_max = st.get_clip_values(None,BGR)
	elif(args.clipping_type=='ImageStyle'):
		BGR=False
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)
	elif(args.clipping_type=='ImageStyleBGR'):
		BGR = True
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)

	if(args.plot):
		plt.ion()
		st.plot_image_with_postprocess(args,image_content.copy(),"Content Image")
		st.plot_image_with_postprocess(args,image_style.copy(),"Style Image")
		fig = None # initialization for later

	# TODO add something that reshape the image
	t1 = time.time()
	pooling_type = args.pooling_type
	padding = args.padding
	vgg_layers = st.get_vgg_layers(args.vgg_name)

	# Precomputation Phase :
	if padding=='VALIDTRUE':
		net = net_preloaded_VALIDTRUE(vgg_layers,image_content,pooling_type) 
	else:
		M_dict = st.get_M_dict(image_h,image_w)
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		net = st.net_preloaded(vgg_layers, image_content,pooling_type,padding)

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


		## We use a perfect result as input to your networks
		image_style_name_split = args.style_img_name.split('_')
		image_style_name_split[-1] = '3'
		content_img_name_perfect2 = "_".join(image_style_name_split)
		image_style_name_split[-1] = 'output_printEvery_' + str(args.print_iter)
		output_img_name = "_".join(image_style_name_split)
		output_image_path = args.img_output_folder + output_img_name + args.img_ext
		init_img = st.load_img(args,content_img_name_perfect2)
		style_layers = list(zip(args.style_layers,args.style_layer_weights))
		if padding=='VALIDTRUE':
			loss_total,style_losses_tab =  style_losses_VALIDTRUE(sess, net, image_style,style_layers)
		else:
			loss_total,style_losses_tab =  st.style_losses(sess, net, dict_gram,M_dict,style_layers)
		list_loss = style_losses_tab
		list_loss_name = ['conv1_1','pool1','pool2','pool3','pool4']

		# Preparation of the assignation operation
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		placeholder_clip = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)
		clip_op = tf.clip_by_value(placeholder_clip,clip_value_min=np.mean(clip_value_min),clip_value_max=np.mean(clip_value_max),name="Clip") # The np.mean is a necessity in the case whe got the BGR values TODO : need to change all that

		if(args.verbose): print("init loss total")

		if(args.optimizer=='adam'): # Gradient Descent with ADAM algo
			optimizer = tf.train.AdamOptimizer(args.learning_rate)
		elif(args.optimizer=='GD'): # Gradient Descente
			if((args.learning_rate > 1) and (args.verbose)): print("We recommande you to use a smaller value of learning rate when using the GD algo")
			optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)

		if((args.optimizer=='GD') or (args.optimizer=='adam')):
			print("Not implemented yet !")
			#train = optimizer.minimize(loss_total)

			#sess.run(tf.global_variables_initializer())
			#sess.run(assign_op, {placeholder: init_img})

			#sess.graph.finalize() # To test if the graph is correct
			#if(args.verbose): print("sess.graph.finalize()")

			#t3 = time.time()
			#if(args.verbose): print("sess Adam initialized after ",t3-t2," s")
			## turn on interactive mode
			#if(args.verbose): print("loss before optimization")
			#if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
			#for i in range(args.max_iter):
				#if(i%args.print_iter==0):
					#if(args.tf_profiler):
						#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
						#run_metadata = tf.RunMetadata()
						#sess.run(train,options=run_options, run_metadata=run_metadata)
						## Create the Timeline object, and write it to a json
						#tl = timeline.Timeline(run_metadata.step_stats)
						#ctf = tl.generate_chrome_trace_format()
						#if(args.verbose): print("Time Line generated")
						#nameFile = 'timeline'+str(i)+'.json'
						#with open(nameFile, 'w') as f:
							#if(args.verbose): print("Save Json tracking")
							#f.write(ctf)
							## Read with chrome://tracing
					#else:
						#t3 =  time.time()
						#sess.run(train)
						#t4 = time.time()
						#result_img = sess.run(net['input'])
						#if(args.clip_var==1): # Clipping the variable
							#cliptensor = sess.run(clip_op,{placeholder_clip: result_img})
							#sess.run(assign_op, {placeholder: cliptensor})
						#if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
						#if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
						#if(args.plot): fig = st.plot_image_with_postprocess(args,result_img,"Intermediate Image",fig)
						#result_img_postproc = st.postprocess(result_img)
						#scipy.misc.toimage(result_img_postproc).save(output_image_path)
				#else:
					## Just training
					#sess.run(train)
					#if(args.clip_var==1): # Clipping the variable
						#result_img = sess.run(net['input'])
						#cliptensor = sess.run(clip_op,{placeholder_clip: result_img})
						#sess.run(assign_op, {placeholder: cliptensor})
		elif(args.optimizer=='lbfgs'):
			# TODO : be able to detect of print_iter > max_iter and deal with it
			nb_iter = args.max_iter  // args.print_iter
			max_iterations_local = args.max_iter // nb_iter
			if(args.verbose): print("Start LBFGS optim with a print each ",max_iterations_local," iterations")
			optimizer_kwargs = {'maxiter': max_iterations_local,'maxcor': args.maxcor}
			# To solve the non retro compatibility of Tensorflow !
			if(tf.__version__ >= '1.3'):
				bnds = st.get_lbfgs_bnds(init_img,clip_value_min,clip_value_max,BGR)
				trainable_variables = tf.trainable_variables()[0]
				var_to_bounds = {trainable_variables: bnds}
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,var_to_bounds=var_to_bounds,
					method='L-BFGS-B',options=optimizer_kwargs)
			else:
				bnds = st.get_lbfgs_bnds_tf_1_2(init_img,clip_value_min,clip_value_max,BGR)
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
					method='L-BFGS-B',options=optimizer_kwargs)
			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})

			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()")

			if(args.verbose): print("loss before optimization")
			if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
			list_loss_eval_total = np.zeros((nb_iter+1,len(list_loss)))
			list_num_iter = np.linspace(0,args.max_iter, num=nb_iter+1,endpoint=True)
			list_loss_eval = compute_loss_tab(sess,list_loss)
			list_loss_eval_total[0,:] = list_loss_eval

			for i in range(nb_iter):
				t3 =  time.time()
				optimizer.minimize(sess)
				t4 = time.time()
				if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
				if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
				result_img = sess.run(net['input'])
				list_loss_eval = compute_loss_tab(sess,list_loss)
				list_loss_eval_total[i+1,:] = list_loss_eval
				# To test the bounds with the algo interface with scipy lbfgs
				if(args.verbose) and test_bnds: print_bnds(result_img,image_style,clip_value_max,clip_value_min)

				if(args.plot): fig = st.plot_image_with_postprocess(args,result_img.copy(),"Intermediate Image",fig)
				result_img_postproc = st.postprocess(result_img)
				scipy.misc.imsave(output_image_path,result_img_postproc)

		# The last iterations are not made
		# The End : save the resulting image
		result_img = sess.run(net['input'])

		if(args.plot): st.plot_image_with_postprocess(args,result_img.copy(),"Final Image",fig)
		result_img_postproc = st.postprocess(result_img)
		scipy.misc.toimage(result_img_postproc).save(output_image_path)
		if args.HistoMatching:
			# Histogram Matching
			if(args.verbose): print("Histogram Matching before saving")
			result_img_postproc = Misc.histogram_matching(result_img_postproc, st.postprocess(image_style))
			output_image_path_hist = args.img_output_folder + args.output_img_name+'_hist' +args.img_ext
			scipy.misc.toimage(result_img_postproc).save(output_image_path_hist)


	except:
		if(args.verbose): print("Error, in the lbfgs case the image can be strange and incorrect")
		result_img = sess.run(net['input'])
		result_img_postproc = st.postprocess(result_img)
		output_image_path_error = args.img_output_folder + args.output_img_name+'_error' +args.img_ext
		scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		# In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
		raise
	finally:
		sess.close()
		if(args.verbose):
			print("Close Sess")

	plt.ion() # Interactive Mode
	plt.figure()
	plt.figure()
	print(style_layers)
	print(list_num_iter)
	for i, couple in enumerate(style_layers):
		layer,_ =couple
		plt.semilogy(list_num_iter,list_loss_eval_total[:,i], label=layer)
	plt.legend()
	plt.title("Comparison of the different layers importance")
	input("Press enter to end and close all")

def texture_grad_originArtefacts(args):
	"""
	Goal of this function is to plot the first gradient and then find 
	the kernel that provoke those artefacts
	 """
	plt.ion()
	image_content = st.load_img(args,args.content_img_name)
	image_style = st.load_img(args,args.style_img_name)
	_,image_h, image_w, number_of_channels = image_content.shape

	if(args.clipping_type=='ImageNet'):
		BGR=False
		clip_value_min,clip_value_max = st.get_clip_values(None,BGR)
	elif(args.clipping_type=='ImageStyle'):
		BGR=False
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)
	elif(args.clipping_type=='ImageStyleBGR'):
		BGR = True
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)
	pooling_type = args.pooling_type
	padding = args.padding
	vgg_layers = st.get_vgg_layers(args.vgg_name)

	# Precomputation Phase :
	if padding=='VALIDTRUE':
		net = net_preloaded_VALIDTRUE(vgg_layers,image_content,pooling_type) 
	else:
		M_dict = st.get_M_dict(image_h,image_w)
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		net = st.net_preloaded(vgg_layers, image_content,pooling_type,padding)
	try:
		config = tf.ConfigProto()
		if(args.gpu_frac <= 0.):
			config.gpu_options.allow_growth = True
			if args.verbose: print("Memory Growth")
		elif(args.gpu_frac <= 1.):
			config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
			if args.verbose: print("Becareful args.gpu_frac = ",args.gpu_frac,"It may cause problem if the value is superior to the available memory place.")
		sess = tf.Session(config=config)


		## We use a perfect result as input to your networks
		image_style_name_split = args.style_img_name.split('_')
		image_style_name_split[-1] = '3'
		content_img_name_perfect2 = "_".join(image_style_name_split)
		image_style_name_split[-1] = 'output'
		output_img_name = "_".join(image_style_name_split)
		output_image_path = args.img_output_folder + output_img_name + args.img_ext

		init_img = st.load_img(args,content_img_name_perfect2)

		st.plot_image_with_postprocess(args,image_style.copy(),"Style Image")
		st.plot_image_with_postprocess(args,init_img.copy(),"Initialisation Image")

		style_layers = list(zip(args.style_layers,args.style_layer_weights))
		if padding=='VALIDTRUE':
			loss_total,style_losses_tab =  style_losses_VALIDTRUE(sess, net, image_style,style_layers)
		else:
			loss_total,style_losses_tab =  st.style_losses(sess, net, dict_gram,M_dict,style_layers)
		list_loss = style_losses_tab
		list_loss_name = ['conv1_1','pool1','pool2','pool3','pool4']

		# Preparation of the assignation operation
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)

		# Gradients
		variable = tf.trainable_variables()
		grad_style_loss = tf.gradients(loss_total,variable)
		grads_and_vars =  list(zip(grad_style_loss,variable))
		print(args.optimizer)
		if(args.optimizer=='GD'): # Gradient Descente
			learning_rate = 1
			optimizer = tf.train.GradientDescentOptimizer(learning_rate) # Gradient Descent
			train = optimizer.apply_gradients(grads_and_vars)
		elif(args.optimizer=='adam'):
			args.learning_rate = 10
			optimizer = tf.train.AdamOptimizer(args.learning_rate)
			train = optimizer.apply_gradients(grads_and_vars)
		elif(args.optimizer=='lbfgs'):
			nb_iter = 1
			max_iterations_local = 1
			optimizer_kwargs = {'maxiter': max_iterations_local,'maxcor': args.maxcor}
			if(tf.__version__ >= '1.3'):
				bnds = st.get_lbfgs_bnds(init_img,clip_value_min,clip_value_max,BGR)
				trainable_variables = tf.trainable_variables()[0]
				var_to_bounds = {trainable_variables: bnds}
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,var_to_bounds=var_to_bounds,
					method='L-BFGS-B',options=optimizer_kwargs)
			else:
				bnds = st.get_lbfgs_bnds_tf_1_2(init_img,clip_value_min,clip_value_max,BGR)
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
					method='L-BFGS-B',options=optimizer_kwargs)

		sess.run(tf.global_variables_initializer())
		sess.run(assign_op, {placeholder: init_img})
						
		sess.graph.finalize() # To test if the graph is correct

		grad_style_loss_eval = sess.run(grad_style_loss)

		# Apply the gradient
		if not(args.optimizer=='lbfgs'): 
			sess.run(train)
		else:
			optimizer.minimize(sess)
		result_img = sess.run(net['input'])
		st.plot_image_with_postprocess(args,result_img.copy(),"First Step Image")
		result_img_postproc = st.postprocess(result_img.copy())
		scipy.misc.toimage(result_img_postproc).save(output_image_path)

		grad_version2 = init_img - result_img 

		grad = grad_style_loss_eval[0]
		print(np.max(grad),np.min(grad),np.std(grad))
		plt.figure()
		plt.imshow(grad[0,:,:,:])
		plt.title('Gradient computed by TF')
		norm_grad = LA.norm(grad)
		grad /= norm_grad
		grad_normalized =grad[0,:,:,:]
		print(np.max(grad_normalized),np.min(grad_normalized),np.std(grad_normalized))
		plt.figure()
		plt.imshow(grad_normalized)
		plt.title('Gradient computed by TF normalized')
		
		print(np.max(grad_version2),np.min(grad_version2),np.std(grad_version2))
		plt.figure()
		plt.imshow(grad_version2[0,:,:,:])
		plt.title('Gradient computed by image diff')
		norm_grad = LA.norm(grad_version2)
		grad_version2 /= norm_grad
		grad_version2_normalized =grad_version2[0,:,:,:]
		print(np.max(grad_version2_normalized),np.min(grad_version2_normalized),np.std(grad_version2_normalized))
		plt.figure()
		plt.imshow(grad_version2_normalized)
		plt.title('Gradient computed by image diff normalized')
		
		diff_grad = grad - grad_version2
		plt.figure()
		plt.imshow(diff_grad[0,:,:,:])
		plt.title('Difference of the Gradient')
		
		input("Press enter to end")
	except:
		print("Error")
		raise
	finally:
		print("Close Sess")


def style_transfer_test_hyp(args,saveFirstGrad=True,test_bnds = False):
	"""
	This function is the main core of the program it need args in order to
	set up all the things and run an optimization in order to produce an
	image
	"""
	if(args.verbose and args.img_ext=='.jpg'): print("Be careful you are saving the image in JPEG !")
	image_content = st.load_img(args,args.content_img_name)
	image_style = st.load_img(args,args.style_img_name)
	_,image_h, image_w, number_of_channels = image_content.shape
	M_dict = st.get_M_dict(image_h,image_w)

	if(args.clipping_type=='ImageNet'):
		BGR=False
		clip_value_min,clip_value_max = st.get_clip_values(None,BGR)
	elif(args.clipping_type=='ImageStyle'):
		BGR=False
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)
	elif(args.clipping_type=='ImageStyleBGR'):
		BGR = True
		clip_value_min,clip_value_max = st.get_clip_values(image_style,BGR)

	if(args.plot):
		plt.ion()
		st.plot_image_with_postprocess(args,image_content.copy(),"Content Image")
		st.plot_image_with_postprocess(args,image_style.copy(),"Style Image")
		fig = None # initialization for later

	# TODO add something that reshape the image
	t1 = time.time()
	pooling_type = args.pooling_type
	padding = args.padding
	vgg_layers = st.get_vgg_layers(args.vgg_name)

	# Precomputation Phase :
	if padding=='VALIDTRUE':
		net = net_preloaded_VALIDTRUE(vgg_layers,image_content,pooling_type) 
	else:
		M_dict = st.get_M_dict(image_h,image_w)
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		net = st.net_preloaded(vgg_layers, image_content,pooling_type,padding)

	try:
		config = tf.ConfigProto()
		if(args.gpu_frac <= 0.):
			config.gpu_options.allow_growth = True
			if args.verbose: print("Memory Growth")
		elif(args.gpu_frac <= 1.):
			config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
			if args.verbose: print("Becareful args.gpu_frac = ",args.gpu_frac,"It may cause problem if the value is superior to the available memory place.")
		sess = tf.Session(config=config)


		## We use a perfect result as input to your networks
		image_style_name_split = args.style_img_name.split('_')
		image_style_name_split[-1] = '3'
		content_img_name_perfect2 = "_".join(image_style_name_split)
		image_style_name_split[-1] = 'output'
		output_img_name = "_".join(image_style_name_split)
		output_image_path = args.img_output_folder + output_img_name + args.img_ext
		init_img = st.load_img(args,content_img_name_perfect2)

		#init_img = get_init_img_wrap(args,output_image_path,image_content)

		style_layers = list(zip(args.style_layers,args.style_layer_weights))
		if padding=='VALIDTRUE':
			loss_total,style_losses_tab =  style_losses_VALIDTRUE(sess, net, image_style,style_layers)
		else:
			loss_total,style_losses_tab =  st.style_losses(sess, net, dict_gram,M_dict,style_layers)
		list_loss = style_losses_tab
		list_loss_name = ['conv1_1','pool1','pool2','pool3','pool4']

		# Preparation of the assignation operation
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		placeholder_clip = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)
		clip_op = tf.clip_by_value(placeholder_clip,clip_value_min=np.mean(clip_value_min),clip_value_max=np.mean(clip_value_max),name="Clip") # The np.mean is a necessity in the case whe got the BGR values TODO : need to change all that

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
			if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
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
						if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
						if(args.plot): fig = st.plot_image_with_postprocess(args,result_img,"Intermediate Image",fig)
						result_img_postproc = st.postprocess(result_img)
						scipy.misc.toimage(result_img_postproc).save(output_image_path)
				else:
					# Just training
					sess.run(train)
					if(args.clip_var==1): # Clipping the variable
						result_img = sess.run(net['input'])
						cliptensor = sess.run(clip_op,{placeholder_clip: result_img})
						sess.run(assign_op, {placeholder: cliptensor})
		elif(args.optimizer=='lbfgs'):
			# TODO : be able to detect of print_iter > max_iter and deal with it
			nb_iter = args.max_iter  // args.print_iter
			max_iterations_local = args.max_iter // nb_iter
			if(args.verbose): print("Start LBFGS optim with a print each ",max_iterations_local," iterations")
			optimizer_kwargs = {'maxiter': max_iterations_local,'maxcor': args.maxcor}
			# To solve the non retro compatibility of Tensorflow !
			if(tf.__version__ >= '1.3'):
				bnds = st.get_lbfgs_bnds(init_img,clip_value_min,clip_value_max,BGR)
				trainable_variables = tf.trainable_variables()[0]
				var_to_bounds = {trainable_variables: bnds}
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,var_to_bounds=var_to_bounds,
					method='L-BFGS-B',options=optimizer_kwargs)
			else:
				bnds = st.get_lbfgs_bnds_tf_1_2(init_img,clip_value_min,clip_value_max,BGR)
				optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
					method='L-BFGS-B',options=optimizer_kwargs)
			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})

			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()")

			if(args.verbose): print("loss before optimization")
			if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
			for i in range(nb_iter):
				t3 =  time.time()
				optimizer.minimize(sess)
				t4 = time.time()
				if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
				if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
				result_img = sess.run(net['input'])

				# To test the bounds with the algo interface with scipy lbfgs
				if(args.verbose) and test_bnds: print_bnds(result_img,image_style,clip_value_max,clip_value_min)

				if(args.plot): fig = st.plot_image_with_postprocess(args,result_img.copy(),"Intermediate Image",fig)
				result_img_postproc = st.postprocess(result_img)
				scipy.misc.imsave(output_image_path,result_img_postproc)

		# The last iterations are not made
		# The End : save the resulting image
		result_img = sess.run(net['input'])

		if(args.plot): st.plot_image_with_postprocess(args,result_img.copy(),"Final Image",fig)
		result_img_postproc = st.postprocess(result_img)
		scipy.misc.toimage(result_img_postproc).save(output_image_path)
		if args.HistoMatching:
			# Histogram Matching
			if(args.verbose): print("Histogram Matching before saving")
			result_img_postproc = Misc.histogram_matching(result_img_postproc, st.postprocess(image_style))
			output_image_path_hist = args.img_output_folder + args.output_img_name+'_hist' +args.img_ext
			scipy.misc.toimage(result_img_postproc).save(output_image_path_hist)

	except:
		if(args.verbose): print("Error, in the lbfgs case the image can be strange and incorrect")
		result_img = sess.run(net['input'])
		result_img_postproc = st.postprocess(result_img)
		output_image_path_error = args.img_output_folder + args.output_img_name+'_error' +args.img_ext
		scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		# In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
		raise
	finally:
		sess.close()
		if(args.verbose):
			print("Close Sess")
	if(args.plot): input("Press enter to end and close all")

def main_loss_decomposition():
	parser = get_parser_args()
	#FabricWool0036_2_seamless_S_small_1
	#TexturesCom_BrickSmallBrown0475_1_M_small_1
	image_style_name= "BrickSmallBrown0293_1_S_C224_small_1"
	#image_style_name= "BrickSmallBrown0293_1_small_1"
	img_output_folder = "images_Hyp/"
	img_folder = "images_Hyp/"
	content_img_name  = image_style_name
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 1.0 # TODO add a gaussian noise on the image instead a uniform one
	content_strengh = 0.001
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	padding = 'SAME'
	# In order to set the parameter before run the script
	parser.set_defaults(img_folder=img_folder,style_img_name=image_style_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,padding=padding,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer,maxcor=maxcor,img_output_folder=img_output_folder,
		learning_rate=learning_rate,sampling=sampling)
	args = parser.parse_args()
	texture_syn_with_loss_decomposition(args)

def main_test_hyp():
	parser = get_parser_args()
	#FabricWool0036_2_seamless_S_small_1
	#TexturesCom_BrickSmallBrown0475_1_M_small_1
	image_style_name= "BrickSmallBrown0293_1_S_C224_small_1"
	image_style_name = 'BrickSmallBrown0293_1_small_1'
	img_output_folder = "images_Hyp/"
	img_folder = "images_Hyp/"
	content_img_name  = image_style_name
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 1.0 # TODO add a gaussian noise on the image instead a uniform one
	content_strengh = 0.001
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	padding = 'SAME'
	#padding = 'VALIDTRUE' # TODO change that it doesn't work yet !!! 
	# In order to set the parameter before run the script
	parser.set_defaults(img_folder=img_folder,style_img_name=image_style_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,padding=padding,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer,maxcor=maxcor,img_output_folder=img_output_folder,
		learning_rate=learning_rate,sampling=sampling)
	args = parser.parse_args()
	style_transfer_test_hyp(args)

def main_artefacts_grad():
	parser = get_parser_args()
	#FabricWool0036_2_seamless_S_small_1
	#TexturesCom_BrickSmallBrown0475_1_M_small_1
	image_style_name= "BrickSmallBrown0293_1_S_C224_small_1"
	image_style_name = 'BrickSmallBrown0293_1_small_1'
	img_output_folder = "images_Hyp/"
	img_folder = "images_Hyp/"
	content_img_name  = image_style_name
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 1.0 # TODO add a gaussian noise on the image instead a uniform one
	content_strengh = 0.001
	optimizer = 'lbfgs'
	optimizer = 'adam'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	padding = 'SAME'
	loss = 'texture'
	# In order to set the parameter before run the script
	parser.set_defaults(img_folder=img_folder,style_img_name=image_style_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,padding=padding,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,loss=loss,
		content_strengh=content_strengh,optimizer=optimizer,maxcor=maxcor,img_output_folder=img_output_folder,
		learning_rate=learning_rate,sampling=sampling)
	args = parser.parse_args()
	texture_grad_originArtefacts(args)

if __name__ == '__main__':
	#main_test_hyp()
	#main_loss_decomposition()
	main_artefacts_grad()


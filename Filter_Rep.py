#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 02 2017

The goal of this script is to vizualised the reponse of the filter of the
different convolution of the network

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

#VGG19_LAYERS_INTEREST = (
    #'conv1_1','conv2_1', 'conv3_1'
#)

VGG19_LAYERS_INTEREST = ('conv1_1' ,'conv1_2','conv2_1' ,'conv2_2' ,
	'conv3_1','conv3_2','conv3_3' ,'conv3_4','conv4_1' ,'conv4_2')

#VGG19_LAYERS_INTEREST = {'conv1_1'}

def hist(values,value_range,nbins=100,dtype=dtypes.float32):
	nbins_float = float(nbins)
	# Map tensor values that fall within value_range to [0, 1].
#    scaled_values = math_ops.truediv(values - value_range[0],
#                                     value_range[1] - value_range[0],
#                                     name='scaled_values') # values - value_range[0] / value_range[1] - value_range[0]
	scaled_values = tf.truediv(values - value_range[0],value_range[1] - value_range[0])
	scaled_values =tf.multiply(nbins_float,scaled_values)
	# map tensor values within the open interval value_range to {0,.., nbins-1},
	# values outside the open interval will be zero or less, or nbins or more.
   # indices = math_ops.floor(nbins_float * scaled_values, name='indices')
	indices = tf.floor(scaled_values)
	print(indices)
	print(type(indices))
	histo = indices
	# Clip edge cases (e.g. value = value_range[1]) or "outliers."
	#indices = math_ops.cast(
	#    clip_ops.clip_by_value(indices, 0, nbins_float- 1), dtypes.int32)

	# TODO(langmore) This creates an array of ones to add up and place in the
	# bins.  This is inefficient, so replace when a better Op is available.
	#histo= math_ops.unsorted_segment_sum(array_ops.ones_like(indices, dtype=dtype),indices,nbins)
	return(histo)

def is_square(apositiveint):
	x = apositiveint // 2
	seen = set([x])
	while x * x != apositiveint:
		x = (x + (apositiveint // x)) // 2
		if x in seen: return False
		seen.add(x)
	return True

def plot_and_save(Matrix,path,name=''):
	 Matrix = Matrix[0] # Remove first dim
	 h,w,channels = Matrix.shape
	 df_Matrix = pd.DataFrame(np.reshape(Matrix,(h*w,channels)))
	 len_columns = len(df_Matrix.columns)
	 if(len_columns<6):
		 fig, axes = plt.subplots(1,len_columns)
	 else:
		 if(len_columns%4==0):
			 fig, axes = plt.subplots(len_columns//4, 4)
		 elif(len_columns%3==0):
			 fig, axes = plt.subplots(len_columns//3, 3)
		 elif(len_columns%5==0):
			 fig, axes = plt.subplots(len_columns//5, 5)
		 elif(len_columns%2==0):
			 fig, axes = plt.subplots(len_columns//2, 2)
		 else:
			 j=6
			 while(not(len_columns%j==0)):
				 j += 1
			 fig, axes = plt.subplots(len_columns//j, j)
	 
	 i = 0
	 axes = axes.flatten()
	 for axis in zip(axes):
		 df_Matrix.hist(column = i, bins = 64, ax=axis)
		 i += 1
	 pltname = path+name+'.png'
	 # TODO avoid to Plot some ligne on the screen
	 fig.savefig(pltname, dpi = 1000)

def plot_and_save_pdf(Matrix,path,name=''):
	pltname = path+name+'_hist.pdf'
	pltname_rep = path+name+'_img.pdf'
	pp = PdfPages(pltname)

	Matrix = Matrix[0] # Remove first dim
	h,w,channels = Matrix.shape
	df_Matrix = pd.DataFrame(np.reshape(Matrix,(h*w,channels)))
	len_columns = len(df_Matrix.columns)
	for i in range(len_columns):
		df_Matrix.hist(column = i, bins = 128)
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()

	plt.clf()
	# Result of the convolution 
	pp_img = PdfPages(pltname_rep)
	for i in range(len_columns):
		plt.imshow(Matrix[:,:,i], cmap='gray')
		plt.savefig(pp_img, format='pdf')
		plt.close()
	pp_img.close()


def plot_Rep(args):
	
	directory_path = 'Results/Filter_Rep/'+args.style_img_name+'/' 
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	
	sns.set()
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_style.shape 
	plot_and_save_pdf(image_style,directory_path,'ProcessIm')
	print("Plot initial image")
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	for layer in VGG19_LAYERS:
		a = net[layer].eval(session=sess)
		print(layer,a.shape)
		plot_and_save_pdf(a,directory_path,layer)

def estimate_gennorm(args):
	
	sns.set_style("white")
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	Distrib_Estimation = {}
	dict_pvalue = {}
	alpha = 0.1
	for layer in VGG19_LAYERS_INTEREST:
		print(layer)
		a = net[layer].eval(session=sess)
		a = a[0]
		h,w,number_of_features = a.shape
		a_reshaped = np.reshape(a,(h*w,number_of_features))
		print(h*w)
		Distrib_Estimation[layer] = np.array([])
		dict_pvalue[layer] = []
		for i in range(number_of_features):
			print(i)
			samples = a_reshaped[:,i]
			# This fit is computed by maximizing a log-likelihood function, with
			# penalty applied for samples outside of range of the distribution. The
			# returned answer is not guaranteed to be the globally optimal MLE, it
			# may only be locally optimal, or the optimization may fail altogether.
			beta, loc, scale = stats.gennorm.fit(samples)
			if(len(Distrib_Estimation[layer])==0):
				print("Number of points",len(samples))
				Distrib_Estimation[layer] = np.array([beta,loc,scale])
			else:
				Distrib_Estimation[layer] =  np.vstack((Distrib_Estimation[layer],np.array([beta,loc,scale])))
			# The KS test is only valid for continuous distributions. and with a theoritical distribution
			D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale ))
			dict_pvalue[layer]  += [pvalue]
			if(pvalue > alpha ): #p-value> α
				print(layer,i,pvalue)
				pass
		#print(Distrib_Estimation[layer])
		#print(dict_pvalue[layer])
	return(Distrib_Estimation)

def unpool(value, name='unpool'):
	"""N-dimensional version of the unpooling operation from
	https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

	:param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
	:return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
	"""
	with tf.name_scope(name) as scope:
		sh = value.get_shape().as_list()
		dim = len(sh[1:-1])
		out = (tf.reshape(value, [-1] + sh[-dim:]))
		for i in range(dim, 0, -1):
			out = tf.concat(i, [out, tf.zeros_like(out)])
		out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
		out = tf.reshape(out, out_size, name=scope)
	return(out)

def calculate_output_shape(in_layer, n_kernel, kernel_size, border_mode='same'):
		"""
		Always assumes stride=1
		"""
		in_shape = in_layer.get_shape() # assumes in_shape[0] = None or batch_size
		out_shape = [s for s in in_shape] # copy
		out_shape[-1] = n_kernel # always true
		if border_mode=='same':
				out_shape[1] = in_shape[1]
				out_shape[2] = in_shape[2]
		elif border_mode == 'valid':
				out_shape[1] = tf.to_int32(in_shape[1]+kernel_size - 1)
				out_shape[2] = tf.to_int32(in_shape[2]+kernel_size - 1)
		return(out_shape)

def genTexture(args):
	
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	output_image_path = args.img_folder + args.output_img_name + args.img_ext 
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	Distrib_Estimation = {}
	dict_pvalue = {}
	data = 'data.pkl'
	
	try:
		Distrib_Estimation = pickle.load(open(data, 'rb'))
	except(FileNotFoundError):
		for layer in VGG19_LAYERS_INTEREST:
			print(layer)
			a = net[layer].eval(session=sess)
			a = a[0]
			h,w,number_of_features = a.shape
			a_reshaped = np.reshape(a,(h*w,number_of_features))
			Distrib_Estimation[layer] = np.array([])
			dict_pvalue[layer] = []
			for i in range(number_of_features):
				samples = a_reshaped[:,i]
				# This fit is computed by maximizing a log-likelihood function, with
				# penalty applied for samples outside of range of the distribution. The
				# returned answer is not guaranteed to be the globally optimal MLE, it
				# may only be locally optimal, or the optimization may fail altogether.
				beta, loc, scale = stats.gennorm.fit(samples)
				if(len(Distrib_Estimation[layer])==0):
					print("Number of points",len(samples))
					Distrib_Estimation[layer] = np.array([beta,loc,scale])
				else:
					Distrib_Estimation[layer] =  np.vstack((Distrib_Estimation[layer],np.array([beta,loc,scale]))) 
					# chaque ligne est un channel
				# The KS test is only valid for continuous distributions. and with a theoritical distribution
				D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale ))
				dict_pvalue[layer]  += [pvalue]
		with open(data, 'wb') as output:
			pickle.dump(Distrib_Estimation,output)

	print('End Computation of marginal distrib')
	generative_net = {}
	generative_net['conv3_1_input'] = tf.Variable(np.zeros(net['conv3_1'].shape, dtype=np.float32))
	weights = tf.constant(np.transpose(vgg_layers[10][0][0][2][0][0], (1, 0, 2, 3)))
	bias = -tf.constant(vgg_layers[10][0][0][2][0][1].reshape(-1))
	#print(weights.get_shape()[0],weights.get_shape()[1],weights.get_shape()[2],weights.get_shape()[3])
	#print(calculate_output_shape(generative_net['conv3_1_input'],tf.to_int32(tf.shape(weights)[3]),tf.to_int32(tf.shape(weights)[1])))
	generative_net['conv3_1'] = tf.nn.conv2d_transpose(value=tf.nn.bias_add(generative_net['conv3_1_input'],bias),filter=weights, 
				  output_shape=calculate_output_shape(generative_net['conv3_1_input'],256,3), 
				  strides=(1, 1, 1, 1),    padding='SAME')
	generative_net['pool2'] = unpool(generative_net['conv3_1'])
	# RELU ???????
	weights = tf.constant(np.transpose(vgg_layers[7][0][0][2][0][0], (1, 0, 2, 3)))
	bias = -tf.constant(vgg_layers[7][0][0][2][0][1].reshape(-1))
	generative_net['conv2_2'] = tf.nn.conv2d_transpose(tf.nn.bias_add(generative_net['pool2'],bias),
				  tf.shape(generative_net['pool2']), weights, strides=(1, 1, 1, 1),    padding='SAME')
	weights = tf.constant(np.transpose(vgg_layers[5][0][0][2][0][0], (1, 0, 2, 3)))
	bias = -tf.constant(vgg_layers[5][0][0][2][0][1].reshape(-1))
	generative_net['conv2_1'] = tf.nn.conv2d_transpose(tf.nn.bias_add(generative_net['conv2_2'],bias),
				  tf.shape(generative_net['conv2_2']),weights, strides=(1, 1, 1, 1),    padding='SAME')
	generative_net['pool1'] = unpool(generative_net['conv2_1'])
	weights = tf.constant(np.transpose(vgg_layers[2][0][0][2][0][0], (1, 0, 2, 3)))
	bias = -tf.constant(vgg_layers[2][0][0][2][0][1].reshape(-1))
	generative_net['conv1_2'] = tf.nn.conv2d_transpose(tf.nn.bias_add(generative_net['pool2'],bias), 
				  tf.shape(generative_net['pool1']),weights, strides=(1, 1, 1, 1),    padding='SAME')
	weights = tf.constant(np.transpose(vgg_layers[0][0][0][2][0][0], (1, 0, 2, 3)))
	bias = -tf.constant(vgg_layers[0][0][0][2][0][1].reshape(-1))
	generative_net['conv1_1'] = tf.nn.conv2d_transpose(tf.nn.bias_add(generative_net['conv2_2'],bias),
				  tf.shape(generative_net['conv1_2']).shape,weights, strides=(1, 1, 1, 1),    padding='SAME')
	generative_net['output'] = tf.Variable(np.zeros(image_style.shape).astype('float32'))
	
	# Random draw marginal distribution 
	for layer in VGG19_LAYERS_INTEREST: 
		print(layer)
		a = net[layer].eval(session=sess)
		a = a[0]
		h,w,number_of_features = a.shape
		#number_samples = h*w
		#a_reshaped = np.reshape(a,(h*w,number_of_features))
		distribs = Distrib_Estimation[layer]
		generative_filters_response = np.zeros(net[layer].shape, dtype=np.float32)
		for i in range(number_of_features):
			print(i)
			beta, loc, scale = distribs[i,:]
			r = stats.gennorm.rvs(beta,loc=loc,scale=scale, size=(h,w))
			generative_filters_response[1,:,:,i] = r
		sess.run(net['conv3_1_input'].assign(generative_filters_response))
		
		print('End generative initialisation')    
	result_img = sess.run(net['input'])            
	result_img_postproc = st.postprocess(result_img)            
	scipy.misc.toimage(result_img_postproc).save(output_image_path)

			 
def generateArt(args):
	if args.verbose:
		print("verbosity turned on")
	
	output_image_path = args.img_folder + args.output_img_name +args.img_ext
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_style.shape 

	t1 = time.time()
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # The output image as the same size as the content one
	t2 = time.time()
	if(args.verbose): print("net loaded and gram computation after ",t2-t1," s")

	try:
		sess = tf.Session()
		init_img = st.get_init_noise_img(image_style,1)
		loss_total =  hist_style_loss(sess,net,image_style)
		
		if(args.verbose): print("init loss total")
		print(tf.trainable_variables())
		#optimizer = tf.train.AdamOptimizer(args.learning_rate) # Gradient Descent
		#train = optimizer.minimize(loss_total)
		bnds = st.get_lbfgs_bnds(init_img)
		optimizer_kwargs = {'maxiter': args.max_iter,'iprint': args.print_iter}
		optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds, method='L-BFGS-B',options=optimizer_kwargs)
		sess.run(tf.global_variables_initializer())
		sess.run(net['input'].assign(init_img)) # This line must be after variables initialization ! 
		optimizer.minimize(sess)
		t3 = time.time()
		if(args.verbose): print("sess Adam initialized after ",t3-t2," s")
		if(args.verbose): print("loss before optimization")
		if(args.verbose): print(sess.run(loss_total))
#        for i in range(args.max_iter):
#            if(i%args.print_iter==0):
#                t3 =  time.time()
#                sess.run(train)
#                t4 = time.time()
#                if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
#                if(args.verbose): print(sess.run(loss_total))
#                result_img = sess.run(net['input'])
#                result_img_postproc = st.postprocess(result_img)
#                scipy.misc.toimage(result_img_postproc).save(output_image_path)
#            else:
#                sess.run(train)
	except:
		print("Error")
		result_img = sess.run(net['input'])
		result_img_postproc = st.postprocess(result_img)
		output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
		scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		raise 
	finally:
		if(args.verbose): print("Close Sess")
		sess.close()
	
def hist_style_loss(sess,net,style_img):
	#value_range = [-2000.0,2000.0] # TODO change according to the layer
	value_range = [-2000.0,2000.0] 
	style_value_range = {'conv1_1' : [-200.0,200.0],'conv2_1': [-500.0,500.0],'conv3_1' :  [-2000.0,2000.0] }
	nbins = 2048
	style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
	#style_layers = [('conv1_1',1.)]
	#style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	length_style_layers = float(len(style_layers))
	sess.run(net['input'].assign(style_img))
	style_loss = 0.0
	weight_help_convergence = 10**9
	for layer, weight in style_layers:
		value_range = style_value_range[layer]
		style_loss_layer = 0.0
		a = sess.run(net[layer])
		_,h,w,N = a.shape
		M =h*w
		tf_M = tf.to_int32(M)
		tf_N = tf.to_int32(N)
		a_reshaped = tf.reshape(a,[tf_M,tf_N])
		a_split = tf.unstack(a_reshaped,axis=1)
		x = net[layer]
		#print("x.op",x.op)
		x_reshaped =  tf.reshape(x,[tf_M,tf_N])
		x_split = tf.unstack(x_reshaped,axis=1)
		
		for a_slide,x_slide in zip(a_split,x_split): # N iteration 
			# Descripteur des representations des histogrammes moment d'ordre 1 a N
			#hist_a = hist(a_slide,value_range, nbins=nbins,dtype=tf.float32)
			#hist_x = hist(x_slide,value_range, nbins=nbins,dtype=tf.float32)
			hist_a = tf.histogram_fixed_width(a_slide, value_range, nbins=nbins,dtype=tf.float32)
			hist_x = tf.histogram_fixed_width(x_slide, value_range, nbins=nbins,dtype=tf.float32)
			#hist_a = tf.floor(a_slide)
			#hist_x = tf.floor(x_slide)
			# TODO normalized les histogrammes avant le calcul plutot qu'apres
			#style_loss_layer += tf.to_float(tf.reduce_mean(tf.abs(hist_a- hist_x))) # norm L1
			#style_loss_layer += tf.reduce_mean(tf.pow(hist_a- hist_x,2)) # Norm L2
			style_loss_layer += tf.sqrt(1-tf.reduce_sum(tf.multiply(tf.sqrt(hist_a),tf.sqrt(hist_x))))
			# TODO use bhattacharyya distance
			
		style_loss_layer *= weight * weight_help_convergence  / (2.*tf.to_float(N**2)*tf.to_float(M**2)*length_style_layers)
		style_loss += style_loss_layer
	return(style_loss)

def do_pdf_comparison(args):
	directory_path = 'Results/Rep/'+args.style_img_name+'/' 
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	
	sns.set_style("white")
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_style.shape 
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_style.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(assign_op, {placeholder: image_style})
	sess.graph.finalize()
	for layer in VGG19_LAYERS_INTEREST:
		a = net[layer].eval(session=sess)
		print(layer,a.shape)
		plot_compare_pdf(vgg_layers,a,directory_path,layer)
	
def plot_compare_pdf(vgg_layers,Matrix,path,name):
	number_img_large_tab = {'conv1_1' : 1,'conv1_2' : 4,'conv2_1' : 4,'conv2_2' : 8,
	'conv3_1' : 8,'conv3_2' : 8,'conv3_3' : 16,'conv3_4' : 16,'conv4_1' : 16,
	'conv4_2' : 16,'conv4_3' : 16,'conv4_4' : 16,'conv5_1' : 16,'conv5_2' : 16,
	'conv5_3' : 16,'conv5_4' : 16}
	pltname = path+name+'_comp.pdf'
	pp = PdfPages(pltname)
	Matrix = Matrix[0] # Remove first dim
	h,w,channels = Matrix.shape
	Matrix_reshaped = np.reshape(Matrix,(h*w,channels))
	df_Matrix = pd.DataFrame(Matrix_reshaped)
	len_columns = len(df_Matrix.columns)
	index_in_vgg = VGG19_LAYERS_INDICES[name]
	kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
	#  A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
	print(kernels.shape)
	bias = vgg_layers[index_in_vgg][0][0][2][0][1]
	print(bias.shape)
	#len_columns = 1
	input_kernel = kernels.shape[2]
	alpha=0.75
	#cmkernel = 'gray'
	cmImg =  'jet'
	cmkernel = plt.get_cmap('hot')
	for i in range(len_columns):
		#print("Features",i)
		# For each feature
		f = plt.figure()
		gs0 = gridspec.GridSpec(1,3, width_ratios=[0.05,4,4]) # 2 columns
		axcm = plt.subplot(gs0[0])
		number_img_large = number_img_large_tab[name]
		if(not(name=='conv1_1')):
			gs00 = gridspec.GridSpecFromSubplotSpec(input_kernel//number_img_large, number_img_large, subplot_spec=gs0[1])
			axes = []
			for j in range(input_kernel):
				ax = plt.subplot(gs00[j])
				axes += [ax]
			kernel = kernels[:,:,:,i]
			mean_kernel = np.mean(kernel)
			bias_i = bias[i,0]
			j = 0
			vmin = np.min(kernel)
			vmax = np.max(kernel)
			for ax in axes:
				im = ax.matshow(kernel[:,:,j],cmap=cmkernel,alpha=alpha,vmin=vmin, vmax=vmax)
				ax.axis('off')
				j += 1
		else:
			gs00 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[1])
			axes = []
			for j in range(input_kernel):
				ax = plt.subplot(gs00[j])
				axes += [ax]
			kernel = kernels[:,:,:,i]
			mean_kernel = np.mean(kernel)
			bias_i = bias[i,0]
			j = 0
			vmin = np.min(kernel)
			vmax = np.max(kernel)
			for ax in axes:
				im = ax.matshow(kernel[:,:,j],cmap=cmkernel,alpha=alpha,vmin=vmin, vmax=vmax)
				ax.axis('off')
				j += 1
			ax0 = plt.subplot(gs00[3])
			# bgr to rgb
			img = kernel[...,::-1]
			#img = Image.fromarray(img, 'RGB')
			#img = exposure.rescale_intensity(img, in_range='uint8')
			img -= np.min(img) 
			img /= (np.max(img)/255.)
			img = np.floor(img).astype('uint8')
			ax0.imshow(img)
			ax0.axis('off')
			ax0.set_title('Color Kernel')
		plt.colorbar(im, cax=axcm)
		#plt.colorbar(im, cax=axes[-1])
		#f.subplots_adjust(right=0.8)
		#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		#f.colorbar(im, cax=cbar_ax)
		#plt.colorbar(im, cax=axes)
		gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[2],height_ratios=[4,3])
		ax1 = plt.subplot(gs01[1])
		ax2 = plt.subplot(gs01[0])
		samples = Matrix_reshaped[:,i]
		beta, loc, scale = stats.gennorm.fit(samples)
		D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale )) # 1-alph = 0.9 
		#D,pvalue = stats.stats.ks_2samp(samples,stats.gennorm.rvs(beta,loc=loc,scale=scale,size=len(samples) ))
		Dcritial = 1.224/math.sqrt(len(samples))
		#print("pvalue",pvalue)
		df_Matrix.hist(column = i, bins = 128,ax=ax1,normed=True, histtype='stepfilled', alpha=1) 
		x = np.linspace(stats.gennorm.ppf(0.005, beta, loc, scale),
                stats.gennorm.ppf(0.995, beta, loc, scale), 128)
		ax1.plot(x, stats.gennorm.pdf(x, beta, loc, scale ),'r-',alpha=0.4, label='gennorm pdf')
		#ax1.legend(loc='best', frameon=False)
		#textstr = '$\mu=%.2f$\n$\mathrm{scale}=%.2f$\n$beta=%.2f$ \n $\mathrm{D}=%.4f$ \n $\mathrm{Dcri}=%.4f$ '%(loc,scale,beta,D,Dcritial)
		textstr = '$\mu=%.2f$\n$\mathrm{scale}=%.2f$\n$beta=%.2f$\n$\mathrm{pvalue}=%.4f$'%(loc,scale,beta,pvalue)
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=6,
			verticalalignment='top', bbox=props)
		ax2.matshow(Matrix[:,:,i], cmap=cmImg)
		ax2.axis('off')
		titre = 'Kernel {} with mean = {:.2e} in range [{:.2e},{:.2e}] and bias = {:.2e}'.format(i,mean_kernel,vmin,vmax,bias_i)
		plt.suptitle(titre)
		#gs0.tight_layout(f)
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()
	plt.clf()
		

	

def main_plot():
	parser = get_parser_args()
	style_img_name = "StarryNight"
	#style_img_name = "Louvre_Big"
	parser.set_defaults(style_img_name=style_img_name)
	args = parser.parse_args()
	plot_Rep(args)
	
def main_distrib():
	parser = get_parser_args()
	style_img_name = "StarryNight"
	parser.set_defaults(style_img_name=style_img_name)
	args = parser.parse_args()
	estimate_gennorm(args)
	


if __name__ == '__main__':
	parser = get_parser_args()
	style_img_name = "StarryNight"
	output_img_name = "Gen"
	max_iter = 10
	print_iter = 1
	parser.set_defaults(style_img_name=style_img_name,output_img_name=output_img_name,
						max_iter=max_iter,    print_iter=print_iter)
	args = parser.parse_args()
	do_pdf_comparison(args)
	

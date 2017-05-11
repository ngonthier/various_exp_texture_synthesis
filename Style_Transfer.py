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

def net_preloaded(vgg_layers, input_image,pooling_type='avg'):
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
			current = _conv_layer(current, kernels, bias,name) 
			# Update the  variable named current to have the right size
		elif(kind == 'relu'):
			current = tf.nn.relu(current,name=name)
		elif(kind == 'pool'):
			current = _pool_layer(current,name)
		net[name] = current

	assert len(net) == len(VGG19_LAYERS) +1 # Test if the length is right 
	return(net)

def _conv_layer(input, weights, bias,name):
	"""
	This function create a conv2d with the already known weight and bias
	
	conv2d :
	Computes a 2-D convolution given 4-D input and filter tensors.
	input: A Tensor. Must be one of the following types: half, float32, float64
	Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
	a filter / kernel tensor of shape 
	[filter_height, filter_width, in_channels, out_channels]
	"""
	conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
			padding='SAME',name=name)
	# We need to impose the weights as constant in order to avoid their modification
	# when we will perform the optimization
	return(tf.nn.bias_add(conv, bias))


def _pool_layer(input,name,pooling_type='avg'):
	"""
	Average pooling on windows 2*2 with stride of 2
	input is a 4D Tensor of shape [batch, height, width, channels]
	Each pooling op uses rectangular windows of size ksize separated by offset 
	strides in the avg_pool function 
	"""
	if pooling_type == 'avg':
		pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding='SAME',name=name) 
	elif pooling_type == 'max':
		pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding='SAME',name=name) 
	return(pool)

def sum_content_losses(sess, net, dict_features_repr):
	"""
	Compute the content term of the loss function
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of the content image representation thanks to the net
	"""
	content_layers = [('conv4_2',1.)]
	length_content_layers = float(len(content_layers))
	weight_help_convergence = 10**(5) 
	content_loss = 0
	for layer, weight in content_layers:
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		content_loss +=  tf.nn.l2_loss(tf.subtract(P,F))* (weight*weight_help_convergence/length_content_layers)
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
	#style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
	style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
	#style_layers = [('conv1_1',1.)]
	# TODO : be able to choose more quickly the different parameters
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	# Info for the vgg19
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 2*10**(9) # This wight come from a paper of Gatys
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
 
def get_Gram_matrix(vgg_layers,image_style,pooling_type):
	"""
	Computation of all the Gram matrices from one image thanks to the 
	vgg_layers
	"""
	net = net_preloaded(vgg_layers, image_style,pooling_type) # net for the style image
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
		 
def get_features_repr(vgg_layers,image_content,pooling_type):
	"""
	Compute the image content representation values according to the vgg
	19 net
	"""
	net = net_preloaded(vgg_layers, image_content,pooling_type) # net for the content image
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
	
	img = img[...,::-1] # rgb to bgr
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
	
def print_loss(sess,loss_total,content_loss,style_loss):
	loss_total_tmp = sess.run(loss_total)
	content_loss_tmp = sess.run(content_loss)
	style_loss_tmp = sess.run(style_loss)
	print("Total loss = ",loss_total_tmp," Content loss = ",content_loss_tmp," Style loss = ",style_loss_tmp)

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

def get_Gram_matrix_wrap(args,vgg_layers,image_style):
	_,image_h_art, image_w_art, _ = image_style.shape
	data_style_path = args.data_folder + "gram_"+args.style_img_name+"_"+str(image_h_art)+"_"+str(image_w_art)+".pkl"
	try:
		dict_gram = pickle.load(open(data_style_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The Gram Matrices doesn't exist, we will generate them.")
		dict_gram = get_Gram_matrix(vgg_layers,image_style,pooling_type)
		with open(data_style_path, 'wb') as output_gram_pkl:
			pickle.dump(dict_gram,output_gram_pkl)
		if(args.verbose): print("Pickle dumped")
	return(dict_gram)

def get_features_repr_wrap(args,vgg_layers,image_content):
	_,image_h, image_w, number_of_channels = image_content.shape 
	data_content_path = args.data_folder +args.content_img_name+"_"+str(image_h)+"_"+str(image_w)+".pkl"
	try:
		dict_features_repr = pickle.load(open(data_content_path, 'rb'))
	except(FileNotFoundError):
		if(args.verbose): print("The dictionnary of features representation of content image doesn't exist, we will generate it.")
		dict_features_repr = get_features_repr(vgg_layers,image_content,pooling_type)
		with open(data_content_path, 'wb') as output_content_pkl:
			pickle.dump(dict_features_repr,output_content_pkl)
		if(args.verbose): print("Pickle dumped")
		
	return(dict_features_repr)


def style_transfer(args,pooling_type='avg'):
	if args.verbose:
		print("verbosity turned on")
	
	output_image_path = args.img_folder + args.output_img_name +args.img_ext
	image_content_path = args.img_folder + args.content_img_name +args.img_ext
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_content = preprocess(scipy.misc.imread(image_content_path).astype('float32')) # Float between 0 and 255
	image_style = preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h, image_w, number_of_channels = image_content.shape 
	M_dict = get_M_dict(image_h,image_w)
	#print("Content")
	#plt.figure()
	#plt.imshow(postprocess(image_content))
	#plt.show()
	#print("Style")
	#plt.figure()
	#plt.imshow(postprocess(image_style))
	#plt.show()
	# TODO add something that reshape the image 
	# TODO : be able to have two different size for the image
	t1 = time.time()

	vgg_layers = get_vgg_layers()
	
	# Precomputation Phase :
	dict_gram = get_Gram_matrix_wrap(args,vgg_layers,image_style)
	dict_features_repr = get_features_repr_wrap(args,vgg_layers,image_content)

	net = net_preloaded(vgg_layers, image_content,pooling_type) # The output image as the same size as the content one
	
	t2 = time.time()
	if(args.verbose): print("net loaded and gram computation after ",t2-t1," s")

	try:
		sess = tf.Session()
		clip_var = True
		if(not(args.start_from_noise)):
			try:
				init_img = preprocess(scipy.misc.imread(output_image_path).astype('float32'))
			except(FileNotFoundError):
				if(args.verbose): print("Former image not found, use of white noise mixed with the content image as initialization image")
				# White noise that we use at the beginning of the optimization
				init_img = get_init_noise_img(image_content,args.init_noise_ratio)
		else:
			init_img = get_init_noise_img(image_content,args.init_noise_ratio)

		#
		# TODO add a plot mode ! 
		#noise_imgS = postprocess(noise_img)
		#plt.figure()
		#plt.imshow(noise_imgS)
		#plt.show()
		
		# Propose different way to compute the lossses 
		style_loss = sum_style_losses(sess,net,dict_gram,M_dict)
		content_loss = args.content_strengh * sum_content_losses(sess, net, dict_features_repr) # alpha/Beta ratio 
		loss_total =  content_loss + style_loss
		
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
			if(args.verbose): print_loss(sess,loss_total,content_loss,style_loss)
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
						if(args.verbose): print_loss(sess,loss_total,content_loss,style_loss)
						
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
			if(args.verbose): print("Start LBFGS optim")
			nb_iter = args.max_iter  // args.print_iter
			max_iterations_local = args.max_iter // nb_iter
			optimizer_kwargs = {'maxiter': max_iterations_local}
			optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds, method='L-BFGS-B',options=optimizer_kwargs)			
			
			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})
						
			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()") 
			
			if(args.verbose): print("loss before optimization")
			if(args.verbose): print_loss(sess,loss_total,content_loss,style_loss)
			for i in range(nb_iter):
				t3 =  time.time()
				optimizer.minimize(sess)
				t4 = time.time()
				if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
				if(args.verbose): print_loss(sess,loss_total,content_loss,style_loss)
				result_img = sess.run(net['input'])
				result_img_postproc = postprocess(result_img)
				scipy.misc.toimage(result_img_postproc).save(output_image_path)
			
		# The last iterations are not made
		# The End : save the resulting image
		result_img = sess.run(net['input'])
		result_img_postproc = postprocess(result_img)
		scipy.misc.toimage(result_img_postproc).save(output_image_path)
		#plt.imshow(result_img)
		#plt.show()
		
		
	except:
		if(args.verbose): print("Error, in the lbfgs case the image can be stranger and incorrect")
		result_img = sess.run(net['input'])
		result_img_postproc = postprocess(result_img)
		output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
		scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		# In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
		raise 
	finally:
		if(args.verbose): print("Close Sess")
		sess.close()

def main():
	#global args
	parser = get_parser_args()
	args = parser.parse_args()
	style_transfer(args)

def main_with_option():
	parser = get_parser_args()
	style_img_name = "StarryNight"
	content_img_name = "Louvre"
	max_iter = 2
	print_iter = 1
	start_from_noise = 1 # True
	init_noise_ratio = 0.1
	content_strengh = 0.001
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	# In order to set the parameter before run the script
	parser.set_defaults(style_img_name=style_img_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer,
		learning_rate=learning_rate)
	args = parser.parse_args()
	style_transfer(args)

if __name__ == '__main__':
	main_with_option()

	
	

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 3 July

The goal of this script is to code the Style Transfer Algorithm 

Inspired from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
and https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb

The goal of this script is to test with a stride = 3 

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
#style_layers = [('input',1),('conv1_1',1),('conv2_1',1),('conv3_1',1)]
#style_layers = [('conv1_1',1),('conv2_1',1),('conv3_1',1)]
#style_layers = [('conv1_1',1)]
#style_layers = [('relu1_1',1),('relu2_1',1),('relu3_1',1)]
#style_layers = [('conv1_1',1),('conv2_1',1),('conv3_1',1),('conv4_1',1),('conv5_1',1)]
style_layers = [('conv1_1',1),('pool1',1),('pool2',1),('pool3',1),('pool4',1)]
#style_layers = [('conv3_1',1)]

style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
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
	## creation of the rotation image
	#temp = current
	#for i in range(3):
		#temp = tf.image.rot90(temp[0], k=1+i)
		#temp = temp[np.newaxis,:,:,:]
		#current = tf.concat([current,temp],axis=0)
	
	#print(current.get_shape())

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
	stride = 1 # TODO try with stride = 3
	if(padding=='SAME'):
		conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
			padding=padding,name=name)
	elif(padding=='VALID'):
		input = get_img_2pixels_more(input)
		conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
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
	stride_pool = 2
	if(padding== 'VALID'): # Test if paire ou impaire !!! 
		_,h,w,_ = input.shape
		if not(h%2==0):
			input = tf.concat([input,input[:,0:2,:,:]],axis=1)
		if not(w%2==0):
			input = tf.concat([input,input[:,:,0:2,:]],axis=2)
	if pooling_type == 'avg':
		pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
				padding=padding,name=name) 
	elif pooling_type == 'max':
		pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
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

def sum_style_losses(sess, net, image_style):
	"""
	Compute the style term of the loss function with Gram Matrix from the
	Gatys Paper
	Input : 
	- the tensforflow session sess
	- the vgg19 net
	- the dictionnary of Gram Matrices
	- the dictionnary of the size of the image content through the net
	"""
	# Info for the vgg19 
	# Tu travailles ici 
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 10**(9) # This wight come from a paper of Gatys
	# Because the function is pretty flat 
	total_style_loss = 0
	#k = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32, seed=None, name=None) # Random Biased
	#image_style = tf.image.rot90(image_style[0], k=k[0], name=None)
	#image_style = image_style[np.newaxis,:,:,:]
	#image_style = tf.contrib.image.rotate(image_style,angles=k)
	sess.run(net['input'].assign(image_style))
	for layer, weight in style_layers:
		# For one layer
		x = net[layer]
		a = sess.run(net[layer])
		b,h,w,N = a.shape
		M = tf.to_int32(h*w)
		N = tf.to_int32(N)
		for j in range(b):
			aa = a[j,:,:,:]
			xx = x[j,:,:,:]
			A = gram_matrix(aa,N,M)
			print(layer,sess.run(M),sess.run(N))
			# Get the value of this layer with the generated image
			G = gram_matrix(xx,N,M) # Nota Bene : the Gram matrix is normalized by M
			style_loss = tf.nn.l2_loss(tf.subtract(G,A))  # output = sum(t ** 2) / 2
			style_loss *=  weight * weight_help_convergence  / (2.*(tf.to_float(N)**2)*length_style_layers)
			total_style_loss += style_loss
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
	dict_gram = {}
	net = net_preloaded(vgg_layers, image_style,pooling_type,padding) # net for the style image
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
	M_dict['input'] = M_dict['conv1']
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

def get_init_noise_img(image_content,init_noise_ratio,range_value):
	""" This function return a white noise image for the initialisation 
	this image can be linearly miwed with the image content with a ratio
	"""
	_,image_h, image_w, number_of_channels = image_content.shape 
	low = 127.5 - range_value
	high = 127.5 + range_value
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
	
def get_init_noise_img_gaussian(image_content):
	"""
	Generate an image with a gaussian white noise
	"""
	b,image_h, image_w, number_of_channels = image_content.shape 
	noise_img = np.random.randn(b,image_h, image_w, number_of_channels) 
	# random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1 
	# Doesn't need preprocess because already arond 0 with a small range
	return(noise_img)
	

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
	"""
	Function that regroup different way to create differente value for 
	initial condition 
	"""
	if(not(args.start_from_noise)):
		try:
			init_img = preprocess(scipy.misc.imread(output_image_path).astype('float32'))
		except(FileNotFoundError):
			if(args.verbose): print("Former image not found, use of white noise mixed with the content image as initialization image")
			# White noise that we use at the beginning of the optimization
			init_img = get_init_noise_img(image_content,args.init_noise_ratio,args.init_range)
	elif(args.init =='smooth_grad'):
		if(args.verbose): print("Noisy image generation with a smooth gradient")
		init_img = get_init_noise_img_smooth_grad(image_content) # TODO add a ratio for this kind of initialization also
	elif(args.init=='Gaussian'):
		if(args.verbose): print("Noisy image generation with a Gaussian white noise")
		init_img = get_init_noise_img_gaussian(image_content)
	elif(args.init=='Uniform'):
		if(args.verbose): print("Noisy image generation init_noise_ratio = ",args.init_noise_ratio)
		init_img = get_init_noise_img(image_content,args.init_noise_ratio,args.init_range)
	elif(args.init=='Cst'):
		if(args.verbose): print("Constante image")
		_,image_h, image_w, number_of_channels = image_content.shape 
		noise_img = (127.5*np.ones((image_h, image_w, number_of_channels))).astype('float32')
		init_img = preprocess(noise_img)
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
		style_loss = sum_style_losses(sess, net, image_style)
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
	if('bizarre'  in args.loss) or ('full' in args.loss):
		 autocorrbizarre_loss = loss_autocorrbizarre(sess,net,image_style,M_dict)
		 list_loss +=  [autocorrbizarre_loss]
		 list_loss_name +=  ['autocorrbizarre_loss']   
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
	if('variance'  in args.loss) or ('full' in args.loss):
		 variance_loss = loss_variance(sess,net,image_style,M_dict)
		 list_loss +=  [variance_loss]
		 list_loss_name +=  ['variance_loss']  
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
		 print("Risk to do a Ressource Exhausted error :) ")
		 intercorr_loss = loss_intercorr(sess,net,image_style,M_dict)
		 list_loss +=  [intercorr_loss]
		 list_loss_name +=  ['intercorr_loss']  
	if('current' in args.loss) or ('full' in args.loss): 
		 PhaseImpose_loss = loss_PhaseImpose(sess,net,image_style,M_dict)   
		 list_loss +=  [PhaseImpose_loss]
		 list_loss_name +=  ['PhaseImpose_loss']    
	if('HF' in args.loss) or ('full' in args.loss):
		 HF_loss = loss__HF_filter(sess, net, image_style,M_dict)   
		 list_loss +=  [HF_loss]
		 list_loss_name +=  ['HF_loss'] 
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
	
def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)
	
def style_transfer(args,pooling_type='avg',padding='SAME'):
	if args.verbose:
		tinit = time.time()
		print("verbosity turned on")
		print(args)
	
	args.img_output_folder = args.img_output_folder + args.style_img_name + '/'
	do_mkdir(args.img_output_folder)
	
	output_image_path = args.img_output_folder + args.output_img_name + args.img_ext
	if(args.verbose and args.img_ext=='.jpg'): print("Be careful you are saving the image in JPEG !")
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
			
			init_img = sess.run(net['input'])
			result_img_postproc = postprocess(init_img)
			output_image_path = args.img_output_folder + args.output_img_name + args.img_ext
			scipy.misc.imsave(output_image_path,result_img_postproc)
						
			#sess.graph.finalize() # To test if the graph is correct
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
				output_image_path = args.img_output_folder + args.output_img_name +"_" +str(i) + args.img_ext
				scipy.misc.imsave(output_image_path,result_img_postproc)
				
				
				if(i%10==0): 
					# do A COPY !!!!! 
					loss_total,list_loss,list_loss_name = get_losses(args,sess, net, dict_features_repr,M_dict,image_style,dict_gram,pooling_type,padding) # Change the input image in the net ! 
					loss_total *= 10.0
					print("increase gradient")
					optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
						method='L-BFGS-B',options=optimizer_kwargs)
					sess.run(tf.global_variables_initializer())
					sess.run(assign_op, {placeholder: result_img})
				# Random shifts are applied to the image to blur tile boundaries over multiple iterations
				#sz = 250
				#h, w = result_img.shape[:2]
				#sx, sy = np.random.randint(sz, size=2)
				#img_shift = np.roll(np.roll(result_img[0], sx, 1), sy, 0)
				#img_shift = np.expand_dims(img_shift,axis=0)
				#sess.run(assign_op, {placeholder: img_shift})

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
		output_image_path_error = args.img_output_folder + args.output_img_name+'_error' +args.img_ext
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
	bleu = "bleu"
	orange = "orange"
	#img_output_folder = "images/"
	image_style_name= orange
	content_img_name  = orange
	#content_img_name  = "Louvre"
	max_iter = 200
	print_iter = 1 
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
	
	

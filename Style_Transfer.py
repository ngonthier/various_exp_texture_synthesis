#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:32:49 2017

The goal of this script is to code the Style Transfer Algorithm 

@author: nicolas
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0' # 1 to remove info, 2 to remove warning and 3 for all
import tensorflow
import tensorflow as tf
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import img_as_float, img_as_ubyte
import pickle
import math

try:
    reduce
except NameError:
	from functools import reduce


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

def net_preloaded(vgg_layers, input_image):
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
	# TODO : other way to add the bias ?
	return(tf.nn.bias_add(conv, bias))


def _pool_layer(input,name):
	"""
	Average pooling on windows 2*2 with stride of 2
	input is a 4D Tensor of shape [batch, height, width, channels]
	Each pooling op uses rectangular windows of size ksize separated by offset 
	strides in the avg_pool function 
	"""
	# TODO add max pool
	return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding='SAME',name=name) 

def sum_content_losses(sess, net, dict_features_repr):
	content_layers = [('conv4_2',1.)]
	length_content_layers = float(len(content_layers))
	content_loss = 0
	for layer, weight in content_layers:
		P = tf.constant(dict_features_repr[layer])
		F = net[layer]
		content_loss +=  tf.nn.l2_loss(tf.subtract(P,F))* (weight/length_content_layers)
	return(content_loss)

def sum_style_losses(sess, net, dict_gram,M_dict):
	#style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]
	style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
	#style_layers = [('conv1_1',1.)]
	style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
	# Info for the vgg19
	length_style_layers = float(len(style_layers))
	weight_help_convergence = 10**(9) # TODO change that
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
		style_loss *=  weight * weight_help_convergence  / (2.*(N**2)*(M**2)*length_style_layers)
		total_style_loss += style_loss
	return(total_style_loss)

def gram_matrix(x,N,M):
  """
  Computation of the Gram Matrix for one layer
  
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
  
  return(G)
 
def get_Gram_matrix(vgg_layers,image_style):
	"""
	Computation of all the Gram matrices from one image thanks to the 
	vgg_layers
	"""
	net = net_preloaded(vgg_layers, image_style) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	dict_gram = {}
	for layer in VGG19_LAYERS:
		kind = layer[:4]
		if(kind == 'conv'): 
			a = net[layer]
			_,height,width,N = a.shape
			M = height*width
			A = gram_matrix(a,N,M) #  TODO Need to divided by M ????
			dict_gram[layer] = sess.run(A) # Computation
	sess.close()
	return(dict_gram)        
		 
def get_features_repr(vgg_layers,image_content):
	net = net_preloaded(vgg_layers, image_content) # net for the content image
	sess = tf.Session()
	sess.run(net['input'].assign(image_content))
	dict_features_repr = {}
	for layer in VGG19_LAYERS:
		kind = layer[:4]
		if(kind == 'conv'): 
			P = sess.run(net[layer])
			dict_features_repr[layer] = P # Computation
	sess.close()
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
	
def main():
	# DEBUG, INFO, WARN, ERROR, or FATAL
	tf.logging.set_verbosity(tf.logging.ERROR)
	#plt.ion()
	image_dir_path = 'images/'
	data_dir_path = 'data/'
	image_content_name = 'Louvre'
	image_style_name  = 'StarryNight'
	#image_content_name = 'Orsay'
	#image_style_name  = 'Nymphea'
	output_image_name = 'Pastiche'
	output_image_path = image_dir_path + output_image_name +'.jpg'
	image_content_path = image_dir_path + image_content_name +'.jpg'    
	image_style_path = image_dir_path + image_style_name +'.jpg'
	image_content = scipy.misc.imread(image_content_path).astype('float32') # Float between 0 and 255
	image_style = scipy.misc.imread(image_style_path).astype('float32') 
	image_h, image_w, number_of_channels = image_content.shape
	M_dict = get_M_dict(image_h,image_w)
	image_content = preprocess(image_content)
	image_style = preprocess(image_style)
	#print("Content")
	#plt.figure()
	#plt.imshow(postprocess(image_content))
	#plt.show()
	#print("Style")
	#plt.figure()
	#plt.imshow(postprocess(image_style))
	#plt.show()
	Content_Strengh = 0.001 # alpha/Beta ratio  TODO : change it
	max_iterations = 5
	print_iterations = 1 # Number of iterations between optimizer print statements
	optimizer = 'adam'
	#optimizer = 'lbfgs'    
	# TODO : be able to have two different size for the image
	# TODO : remove mean in preprocessing and add mean in post process
	t1 = time.time()

	
	vgg_layers = get_vgg_layers()
	
	# Precomputation Phase :
	data_style_path = data_dir_path + "gram_"+image_style_name+".pkl"
	try:
		dict_gram = pickle.load(open(data_style_path, 'rb'))
	except(FileNotFoundError):
		print("The Gram Matrices doesn't exist, we will generate them.")
		dict_gram = get_Gram_matrix(vgg_layers,image_style)
		with open(data_style_path, 'wb') as output_gram_pkl:
			pickle.dump(dict_gram,output_gram_pkl)
		print("Pickle dumped")

	data_content_path = data_dir_path +image_content_name+".pkl"
	try:
		dict_features_repr = pickle.load(open(data_content_path, 'rb'))
	except(FileNotFoundError):
		print("The dictionnary of features representation of content image doesn't exist, we will generate it.")
		dict_features_repr = get_features_repr(vgg_layers,image_content)
		with open(data_content_path, 'wb') as output_content_pkl:
			pickle.dump(dict_features_repr,output_content_pkl)
		print("Pickle dumped")


	net = net_preloaded(vgg_layers, image_content) # The output image as the same size as the content one
	t2 = time.time()
	print("net loaded and gram computation after ",t2-t1," s")

	try:
		sess = tf.Session()
		
		try:
			noise_img = scipy.misc.imread(output_image_path).astype('float32')
		except(FileNotFoundError):
			print("Former image not found, use of white noise as initialization image")
			# White noise that we use at the beginning of the optimization
			noise_img = np.random.uniform(0,255, (image_h, image_w, number_of_channels)).astype('float32')
		#noise_img = scipy.misc.imread(image_style_path).astype('float32') 
		#noise_img = scipy.misc.imread(image_content_path).astype('float32')
		init_noise_ratio = 0.75
		noise_img = preprocess(noise_img)
		#noise_img = init_noise_ratio* noise_img + (1.-init_noise_ratio) * image_content
		# TODO add a plot mode ! 
		#noise_imgS = postprocess(noise_img)
		#plt.figure()
		#plt.imshow(noise_imgS)
		#plt.show()
				
		#loss_total = tf.add(tf.multiply(tf.constant(Content_Strengh),sum_content_losses(sess, net, dict_features_repr)),sum_style_losses(sess,net,dict_gram,M_dict))
		loss_total = Content_Strengh * sum_content_losses(sess, net, dict_features_repr) + sum_style_losses(sess,net,dict_gram,M_dict)
		#loss_total = sum_content_losses(sess, net, dict_features_repr)
		#loss_total = sum_style_losses(sess,net,dict_gram,M_dict)
		
		print("init loss total")
				
		# TODO image mixed content image with white noise
		if(optimizer=='adam'):
			learning_rate = 10.0
			optimizer = tf.train.AdamOptimizer(learning_rate) # Gradient Descent
			# TODO function in order to use different optimization function
			train = optimizer.minimize(loss_total)

			sess.run(tf.global_variables_initializer())
			sess.run(net['input'].assign(noise_img)) # This line must be after variables initialization ! 
			t3 = time.time()
			print("sess Adam initialized after ",t3-t2," s")
			# turn on interactive mode
			print("loss before optimization = ",sess.run(loss_total))
			
			for i in range(max_iterations):
				if(i%print_iterations==0):
					t3 =  time.time()
					sess.run(train)
					t4 = time.time()
					print("Iteration ",i, "after ",t4-t3," s")
					print("loss = ",sess.run(loss_total))
					result_img = sess.run(net['input'])
					result_img = postprocess(result_img)
					scipy.misc.toimage(result_img).save(output_image_path)
				else:
					sess.run(train)
		elif(optimizer=='lbfgs'):
			z,height,width,N = noise_img.shape
			bnd_inf = -124*np.ones((z,height,width,N)).flatten() 
			# We need to flatten the array in order to use it in the LBFGS algo
			bnd_sup = 152*np.ones((z,height,width,N)).flatten()
			bnds = np.stack((bnd_inf, bnd_sup),axis=-1)
			print("Start LBFGS optim")
			t3 =  time.time()
			optimizer_kwargs = {'maxiter': max_iterations,'disp': print_iterations}
			optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds, method='L-BFGS-B',options=optimizer_kwargs)
			# Bounds from [0,255] - [124,103]
			# TODO Add checkpoint
			sess.run(tf.global_variables_initializer())
			sess.run(net['input'].assign(noise_img))
			optimizer.minimize(sess)
			t4 = time.time()
			print("LBFGS optim after ",t4-t3," s")
		# TODO add a remove old image
		result_img = sess.run(net['input'])
		result_img = postprocess(result_img)
		#print(np.min(result_img),np.max(result_img))
		#plt.imshow(result_img)
		#plt.show()
		scipy.misc.toimage(result_img).save(output_image_path)

	except:
		print("Error")
		result_img = sess.run(net['input'])
		print(np.min(result_img),np.max(result_img))
		result_img = postprocess(result_img)
		print(np.min(result_img),np.max(result_img))
		#plt.imshow(result_img)
		#plt.show()
		scipy.misc.toimage(result_img).save(output_image_path)
		raise 
	finally:
		print("Close Sess")
		sess.close()

if __name__ == '__main__':
	main()
	# 1.16 s

	
	
